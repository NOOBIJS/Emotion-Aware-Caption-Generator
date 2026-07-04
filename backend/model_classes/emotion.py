from deepface import DeepFace
import tempfile
import os
import nltk
from nltk import pos_tag, word_tokenize

# Download NLTK data (already done in Dockerfile, but just in case)
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

try:
    nltk.data.find('taggers/averaged_perceptron_tagger')
except LookupError:
    nltk.download('averaged_perceptron_tagger', quiet=True)


def detect_emotion_deepface(image):
    """Detect emotion from image using DeepFace"""
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
            image.save(tmp_file.name)
            tmp_path = tmp_file.name
        
        result = DeepFace.analyze(tmp_path, actions=['emotion'], enforce_detection=False)
        os.unlink(tmp_path)
        
        if isinstance(result, list):
            result = result[0]
        
        emotion = result['dominant_emotion']
        return emotion
    except Exception as e:
        print(f"Error in emotion detection: {e}")
        return "neutral"


def insert_emotion_in_caption(caption, emotion):
    """Insert emotion adjective before first noun"""
    try:
        emotion_adjectives = {
            'happy': 'joyful',
            'sad': 'melancholic',
            'angry': 'angry',
            'surprise': 'surprised',
            'fear': 'fearful',
            'disgust': 'disgusted',
            'neutral': 'calm'
        }
        
        adjective = emotion_adjectives.get(emotion.lower(), emotion.lower())
        tokens = word_tokenize(caption)
        pos_tags = pos_tag(tokens)
        
        for i, (word, tag) in enumerate(pos_tags):
            if tag.startswith('NN'):
                tokens.insert(i, adjective)
                break
        
        return ' '.join(tokens)
    except Exception as e:
        print(f"Error inserting emotion: {e}")
        return f"{emotion} {caption}"


def generate_emotion_aware_caption(image, model, tokenizer, transform, device, max_len=40):
    """Generate emotion-aware caption and return all details"""
    try:
        # Generate base caption
        image_tensor = transform(image).unsqueeze(0).to(device)
        
        token_ids = model.generate(
            image_tensor,
            sos_token=tokenizer.get_vocab()['[BOS]'],
            eos_token=tokenizer.get_vocab()['[EOS]'],
            max_len=max_len
        )
        
        base_caption = tokenizer.decode([t.item() if hasattr(t, 'item') else t for t in token_ids])
        base_caption = base_caption.replace('[BOS]', '').replace('[EOS]', '').replace('[PAD]', '').strip()
        
        # Detect emotion
        emotion = detect_emotion_deepface(image)
        
        # Insert emotion
        emotion_caption = insert_emotion_in_caption(base_caption, emotion)
        
        # Return detailed breakdown
        return {
            "base_caption": base_caption,
            "detected_emotion": emotion,
            "emotion_aware_caption": emotion_caption
        }
    except Exception as e:
        print(f"Error: {e}")
        return {
            "base_caption": "Error generating caption",
            "detected_emotion": "unknown",
            "emotion_aware_caption": "Error generating caption"
        }