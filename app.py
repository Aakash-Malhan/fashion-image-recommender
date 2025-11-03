import os, glob, json, zipfile, shutil, webbrowser
from typing import List, Tuple, Optional
import numpy as np
from PIL import Image, ImageOps
import gradio as gr
from sklearn.metrics.pairwise import cosine_similarity

# ML Models
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing import image as keras_image

# FAISS for fast similarity
import faiss


# CONFIG

DATA_DIR = os.getenv("DATA_DIR", "women_fashion/women fashion")
INDEX_DIR = "index_cache"
ZIP_NAME = "de18b-women-fashion.zip"
IMG_EXT = (".jpg", ".jpeg", ".png", ".webp")

os.makedirs(INDEX_DIR, exist_ok=True)


# ZIP EXTRACT

def extract_zip():
    if os.path.isdir(DATA_DIR): return DATA_DIR
    if not os.path.exists(ZIP_NAME): return DATA_DIR

    with zipfile.ZipFile(ZIP_NAME, "r") as zip_ref:
        zip_ref.extractall("women_fashion")

    for root, _, files in os.walk("women_fashion"):
        if any(f.lower().endswith(IMG_EXT) for f in files):
            return root

    return DATA_DIR

DATA_DIR = extract_zip()


# IMAGE LISTING

def list_images(folder):
    files = []
    for ext in IMG_EXT:
        files.extend(glob.glob(os.path.join(folder, f"*{ext}")))
    return sorted(files)


# IMAGE PROCESSING HELPERS

def preprocess_img(path):
    img = keras_image.load_img(path, target_size=(224,224))
    arr = keras_image.img_to_array(img)
    arr = np.expand_dims(arr, axis=0)
    return preprocess_input(arr)

def preprocess_pil(img):
    img = img.convert("RGB").resize((224,224))
    arr = np.array(img).astype(np.float32)
    arr = np.expand_dims(arr, axis=0)
    return preprocess_input(arr)

def color_hist(path):
    img = Image.open(path).convert("RGB")
    img = img.resize((100,100))
    hsv = img.convert("HSV")
    hist = np.array(hsv.histogram()).astype("float32")
    hist /= hist.sum()
    return hist


# LOAD VGG + FAISS INDEX

base = VGG16(weights="imagenet", include_top=False)
model = Model(inputs=base.input, outputs=base.output)

def extract_feature(img):
    feat = model.predict(img, verbose=0)
    feat = feat.flatten()
    feat /= (np.linalg.norm(feat) + 1e-7)
    return feat

image_paths = list_images(DATA_DIR)

faiss_index = None
features = []
colors = []


def build_index():
    global faiss_index, features, colors

    cached_feats = os.path.join(INDEX_DIR, "vgg.npy")
    cached_paths = os.path.join(INDEX_DIR, "paths.json")
    cached_colors = os.path.join(INDEX_DIR, "colors.npy")

    if os.path.exists(cached_feats) and os.path.exists(cached_paths):
        with open(cached_paths, "r") as f: cached = json.load(f)
        if cached == image_paths:
            features = np.load(cached_feats)
            colors = np.load(cached_colors)
            d = features.shape[1]
            faiss_index = faiss.IndexFlatIP(d)
            faiss_index.add(features)
            return

    feats, cols = [], []
    for p in image_paths:
        arr = preprocess_img(p)
        feats.append(extract_feature(arr))
        cols.append(color_hist(p))

    features = np.vstack(feats).astype("float32")
    colors = np.vstack(cols).astype("float32")

    d = features.shape[1]
    faiss_index = faiss.IndexFlatIP(d)
    faiss_index.add(features)

    np.save(cached_feats, features)
    np.save(cached_colors, colors)
    with open(cached_paths, "w") as f: json.dump(image_paths, f)

build_index()


# RANKING LOGIC

def rank_results(query_feat, query_color, top_k=5, alpha=0.8):
    sim, idxs = faiss_index.search(query_feat.reshape(1,-1), top_k+1)
    idxs = idxs[0][1:]   

    deep_scores = sim[0][1:]
    col_scores = cosine_similarity(query_color.reshape(1,-1), colors[idxs])[0]

    # weighted blend
    final_scores = alpha * deep_scores + (1-alpha) * col_scores
    rank = np.argsort(final_scores)[::-1]

    return [(idxs[i], final_scores[i]) for i in rank]


def explain_match(qpath, rpath):
    """very lightweight visual rule based explanation"""
    def keyword(path, words):
        name = os.path.basename(path).lower()
        return any(w in name for w in words)

    notes = []
    if keyword(qpath, ["sequ"]): notes.append("✨ sequin texture")
    if keyword(rpath, ["sequ"]): notes.append("✨ sequin match")

    if keyword(qpath, ["mini"]): notes.append("short length")
    if keyword(rpath, ["mini"]): notes.append("matched length")

    if keyword(qpath, ["v-neck","v neck"]): notes.append("V-neck style")
    if keyword(rpath, ["v-neck","v neck"]): notes.append("matched neckline")

    if not notes: notes = ["Similar silhouette & color tone"]
    return ", ".join(notes)


# UI LOGIC

def recommend(mode, upload_img, select_img, top_n):
    if mode == "Upload image":
        if upload_img is None: return None, [], "Upload an image."
        arr = preprocess_pil(upload_img)
        qfeat = extract_feature(arr)
        qcolor = color_hist_from_pil(upload_img)
        qpath = "User Input"
    else:
        if not select_img: return None, [], "Select an image first."
        full = os.path.join(DATA_DIR, select_img)
        arr = preprocess_img(full)
        qfeat = extract_feature(arr)
        qcolor = color_hist(full)
        qpath = full
        upload_img = Image.open(full)

    ranked = rank_results(qfeat, qcolor, top_n)
    results = []
    notes = ""

    for idx, score in ranked:
        p = image_paths[idx]
        img = Image.open(p).convert("RGB")
        caption = f"{os.path.basename(p)}\nScore: {score:.2f}\nWhy: {explain_match(qpath, p)}"
        results.append((img, caption))

    return upload_img, results, f"✅ Found {len(results)} matches"


def color_hist_from_pil(img):
    img = img.convert("HSV").resize((100,100))
    hist = np.array(img.histogram()).astype("float32")
    hist /= hist.sum()
    return hist


# REINDEX BUTTON

def rebuild():
    shutil.rmtree(INDEX_DIR, ignore_errors=True)
    os.makedirs(INDEX_DIR, exist_ok=True)
    build_index()
    return gr.update(choices=[os.path.basename(p) for p in image_paths]), "✅ Index rebuilt."


# GRADIO UI

with gr.Blocks(title="Fashion Recommender Pro") as demo:
    gr.Markdown("## 👗 AI Fashion Recommendation System\nEnhanced with **FAISS + Color Awareness + Explanations**")

    mode = gr.Radio(["Upload image", "Pick from dataset"], value="Upload image")
    upload = gr.Image(type="pil", label="Upload Your Dress")
    select = gr.Dropdown([os.path.basename(p) for p in image_paths], label="Or select from dataset")
    top_n = gr.Slider(3,12,value=4,step=1,label="Recommendations")
    run = gr.Button("Recommend")
    re = gr.Button("Rebuild index")

    inp = gr.Image(label="Input")
    gallery = gr.Gallery(label="Matches", columns=4, height=500)
    status = gr.Markdown()

    run.click(recommend, inputs=[mode, upload, select, top_n], outputs=[inp, gallery, status])
    re.click(rebuild, outputs=[select, status])


if __name__ == "__main__":
    demo.launch()
