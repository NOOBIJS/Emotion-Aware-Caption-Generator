from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import torch
from torchvision import transforms
from PIL import Image
import io
import os

from model_classes import TokenizerHF, ImageCaptionModel, generate_emotion_aware_caption

app = FastAPI(title="Emotion-Aware Image Captioning API")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://emotion-caption-frontend.vercel.app/",
        "http://localhost:3000"  # Keep this for local testing
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables
model = None
tokenizer = None
transform = None
device = None


@app.on_event("startup")
async def load_model():
    """Load model on startup"""
    global model, tokenizer, transform, device
    
    print("🚀 Loading model...")
    
    # Set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Download model from HuggingFace Models hub
    checkpoint_path = "image_caption_model.pt"
    
    if not os.path.exists(checkpoint_path):
        print("📥 Downloading 1.91GB model from HuggingFace... (this takes 5-10 mins)")
        import urllib.request
        
        # Progress callback
        def show_progress(block_num, block_size, total_size):
            downloaded = block_num * block_size
            percent = min(downloaded * 100 / total_size, 100)
            print(f"Download progress: {percent:.1f}%", end='\r')
        
        model_url = "https://huggingface.co/HimadriBiswas/emotion-caption-model/resolve/main/image_caption_model.pt"
        urllib.request.urlretrieve(model_url, checkpoint_path, show_progress)
        print("\n✅ Model downloaded!")
    else:
        print("✅ Model already cached!")
    
    # Load checkpoint
    print("Loading checkpoint into memory...")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Update device in config
    checkpoint['model_config']['device'] = device
    checkpoint['model_config']['vit_kwargs']['device'] = device
    checkpoint['model_config']['gpt_kwargs']['device'] = device
    
    # Create model
    print("Creating model architecture...")
    model = ImageCaptionModel(checkpoint['model_config']).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Create tokenizer
    tokenizer = TokenizerHF(
        tokenizer_name="gpt2",
        special_tokens_dict={"bos_token": "[BOS]", "eos_token": "[EOS]", "pad_token": "[PAD]"}
    )
    
    # Create transform
    transform = transforms.Compose([
        transforms.Resize(size=(224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    print("✅ Model loaded successfully!")


@app.get("/")
async def root():
    """Health check endpoint"""
    return {"status": "running", "message": "Emotion-Aware Image Captioning API"}


@app.post("/api/caption")
async def generate_caption(file: UploadFile = File(...)):
    """
    Generate emotion-aware caption for uploaded image
    """
    try:
        # Validate file type
        if not file.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        # Read image
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        
        # Generate emotion-aware caption (returns dict with details)
        result = generate_emotion_aware_caption(
            image=image,
            model=model,
            tokenizer=tokenizer,
            transform=transform,
            device=device,
            max_len=40
        )
        
        return JSONResponse(content=result)
    
    except Exception as e:
        print(f"Error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error generating caption: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)