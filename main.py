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

import threading

model       = None
model_ready = False

def load_model_background():
    global model, class_names, model_ready, supabase

    supabase = create_client(
        os.environ['SUPABASE_URL'],
        os.environ['SUPABASE_KEY']
    )

    print('Downloading model from Hugging Face...')
    model_dir = snapshot_download(
        repo_id   = os.environ['HF_REPO'],
        repo_type = 'model',
        token     = os.environ.get('HF_TOKEN')
    )

    cn_path = os.path.join(model_dir, 'class_names.json')
    if os.path.exists(cn_path):
        with open(cn_path) as f:
            m = json.load(f)
        class_names = [m[str(i)] for i in range(len(m))]

    model = tf.keras.models.load_model(
        os.path.join(model_dir, 'best_model.keras')
    )
    model_ready = True
    print(f'Model ready. Classes: {class_names}')

@app.on_event('startup')
async def startup():
    # Launch model loading in background — port opens immediately
    thread = threading.Thread(target=load_model_background, daemon=True)
    thread.start()

@app.get('/health')
async def health():
    return {'status': 'ok', 'model': model_ready}

@app.post('/predict')
async def predict(...):
    if not model_ready:
        raise HTTPException(status_code=503, detail='Model still loading, try again in 30 seconds')
    # ... rest of predict unchanged

@app.post('/predict')
async def predict(
    file:   UploadFile = File(...),
    name:   str        = Form(...),
    age:    str        = Form(''),
    gender: str        = Form(''),
    email:  str        = Form(''),
):
    contents = await file.read()
    try:
        img = preprocess(contents)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    finally:
        del contents

    import tensorflow as tf
    batch  = tf.constant(np.expand_dims(img, 0))
    probs  = model.predict(batch, verbose=0)[0].tolist()
    result = class_names[int(np.argmax(probs))]
    del img, batch

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

    return {'prediction': result, 'user_id': user_id}
 
# ── History endpoint ──────────────────────────────────────────────
@app.get('/history')
async def history(limit: int = 50):
    res = supabase.table('result_details').select('*').limit(limit).execute()
    return res.data
 
@app.delete('/history/{result_id}')
async def delete_record(result_id: str):
    supabase.table('results').delete().eq('id', result_id).execute()
    return {'ok': True}
 
@app.get('/health')
async def health():
    return {'status': 'ok', 'model': model is not None}
