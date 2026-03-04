import os, json, io
import numpy as np
import cv2
import tensorflow as tf
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from supabase import create_client
from huggingface_hub import snapshot_download
 
app = FastAPI(title='RidgeScan API')
 
# Allow the Flask frontend to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_methods=['*'],
    allow_headers=['*']
)
 
# ── Globals loaded at startup ─────────────────────────────────────
model       = None
class_names = ['A+','A-','AB+','AB-','B+','B-','O+','O-']
supabase    = None
 
@app.on_event('startup')
async def startup():
    global model, class_names, supabase
 
    # Connect to Supabase
    supabase = create_client(
        os.environ['SUPABASE_URL'],
        os.environ['SUPABASE_KEY']
    )
 
    # Download model from Hugging Face Hub
    print('Downloading model from Hugging Face...')
    model_dir = snapshot_download(
        repo_id   = os.environ['HF_REPO'],
        repo_type = 'model',
        token     = os.environ.get('HF_TOKEN')  # needed if repo is private
    )
 
    # Load class names
    cn_path = os.path.join(model_dir, 'class_names.json')
    if os.path.exists(cn_path):
        with open(cn_path) as f:
            m = json.load(f)
        class_names = [m[str(i)] for i in range(len(m))]
 
    # Load model
    model = tf.saved_model.load(model_dir)
    print(f'Model ready. Classes: {class_names}')
 
# ── Preprocessing ─────────────────────────────────────────────────
IMG_SIZE = 300
 
def preprocess(file_bytes: bytes):
    arr = np.frombuffer(file_bytes, np.uint8)     # bytes → array
    img = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE) # decode in RAM
    if img is None:
        raise ValueError('Cannot decode image')
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = np.stack([img, img, img], axis=-1)      # 3-channel
    return img.astype(np.float32)                 # NO /255
 
# ── Predict endpoint ──────────────────────────────────────────────
@app.post('/predict')
async def predict(
    file:   UploadFile = File(...),
    name:   str        = Form(...),
    age:    str        = Form(''),
    gender: str        = Form(''),
    email:  str        = Form(''),
):
    # Read file bytes — image NEVER touches disk
    contents = await file.read()
 
    try:
        img = preprocess(contents)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    finally:
        del contents  # free memory immediately
 
    # Run inference
    batch  = tf.constant(np.expand_dims(img, 0))
    infer  = model.signatures['serving_default']
    output = infer(batch)
    probs  = list(output.values())[-1].numpy()[0]
    result = class_names[int(np.argmax(probs))]
    del img, batch  # free memory
 
    # Save user + result to Supabase
    user_res = supabase.table('users').insert({
        'name':   name,
        'age':    int(age) if age.isdigit() else None,
        'gender': gender or None,
        'email':  email or None,
    }).execute()
 
    user_id = user_res.data[0]['id']
 
    supabase.table('results').insert({
        'user_id':     user_id,
        'blood_group': result
    }).execute()
 
    return { 'prediction': result, 'user_id': user_id }
 
# ── History endpoint ──────────────────────────────────────────────
@app.get('/history')
async def history(limit: int = 50):
    res = supabase.table('result_details')
             .select('*')
             .limit(limit)
             .execute()
    return res.data
 
@app.delete('/history/{result_id}')
async def delete_record(result_id: str):
    supabase.table('results').delete().eq('id', result_id).execute()
    return {'ok': True}
 
@app.get('/health')
async def health():
    return {'status': 'ok', 'model': model is not None}
