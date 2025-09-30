from fastapi import FastAPI, UploadFile, Form, Depends, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from sqlalchemy import text
from database import engine, Base, SessionLocal
import models
import cv2, numpy as np, uuid, os, io, zipfile

# -------------------------
# Setup
# -------------------------
Base.metadata.create_all(bind=engine)
app = FastAPI(title="Image Processing App")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# -------------------------
# DB Dependency
# -------------------------
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# -------------------------
# Helpers
# -------------------------
def read_image(file: UploadFile):
    """Convert uploaded file → OpenCV image."""
    nparr = np.frombuffer(file.file.read(), np.uint8)
    return cv2.imdecode(nparr, cv2.IMREAD_COLOR)


def process_with_filter(img, operation: str):
    """Apply filter to image using OpenCV."""
    if operation == "grayscale":
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    elif operation == "canny":
        edges = cv2.Canny(img, 100, 200)
        return cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    elif operation == "blur":
        return cv2.GaussianBlur(img, (5, 5), 0)
    elif operation == "hsv":
        return cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    elif operation == "hsv_hue":
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        hue = hsv[:, :, 0]
        return cv2.cvtColor(hue, cv2.COLOR_GRAY2BGR)
    elif operation == "hsv_saturation":
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        saturation = hsv[:, :, 1]
        return cv2.cvtColor(saturation, cv2.COLOR_GRAY2BGR)
    elif operation == "hsv_value":
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        value = hsv[:, :, 2]
        return cv2.cvtColor(value, cv2.COLOR_GRAY2BGR)
    
    # BGR Color Channel Operations
    elif operation == "blue_channel_gray":
        B, G, R = cv2.split(img)
        return cv2.cvtColor(B, cv2.COLOR_GRAY2BGR)
    elif operation == "green_channel_gray":
        B, G, R = cv2.split(img)
        return cv2.cvtColor(G, cv2.COLOR_GRAY2BGR)
    elif operation == "red_channel_gray":
        B, G, R = cv2.split(img)
        return cv2.cvtColor(R, cv2.COLOR_GRAY2BGR)
    elif operation == "blue_channel_only":
        B, G, R = cv2.split(img)
        zeros = np.zeros(img.shape[:2], dtype="uint8")
        return cv2.merge([B, zeros, zeros])
    elif operation == "green_channel_only":
        B, G, R = cv2.split(img)
        zeros = np.zeros(img.shape[:2], dtype="uint8")
        return cv2.merge([zeros, G, zeros])
    elif operation == "red_channel_only":
        B, G, R = cv2.split(img)
        zeros = np.zeros(img.shape[:2], dtype="uint8")
        return cv2.merge([zeros, zeros, R])
    elif operation == "blue_boost":
        B, G, R = cv2.split(img)
        B_boosted = np.clip(B.astype(np.int16) + 100, 0, 255).astype(np.uint8)
        return cv2.merge([B_boosted, G, R])
    elif operation == "red_boost":
        B, G, R = cv2.split(img)
        R_boosted = np.clip(R.astype(np.int16) + 100, 0, 255).astype(np.uint8)
        return cv2.merge([B, G, R_boosted])
    elif operation == "green_boost":
        B, G, R = cv2.split(img)
        G_boosted = np.clip(G.astype(np.int16) + 100, 0, 255).astype(np.uint8)
        return cv2.merge([B, G_boosted, R])
    
    # Drawing Operations
    elif operation == "add_diagonal_line":
        img_copy = img.copy()
        height, width = img.shape[:2]
        cv2.line(img_copy, (0, 0), (width-1, height-1), (255, 127, 0), 5)
        return img_copy
    elif operation == "add_rectangle":
        img_copy = img.copy()
        height, width = img.shape[:2]
        rect_w, rect_h = width//3, height//3
        cv2.rectangle(img_copy, (20, 20), (rect_w, rect_h), (127, 50, 127), 10)
        return img_copy
    elif operation == "add_circle":
        img_copy = img.copy()
        height, width = img.shape[:2]
        center_x, center_y = width//2, height//2
        radius = min(width, height) // 6
        cv2.circle(img_copy, (center_x, center_y), radius, (15, 150, 50), -1)
        return img_copy
    elif operation == "add_polygon":
        img_copy = img.copy()
        height, width = img.shape[:2]
        pts = np.array([
            [width//20, height//10], 
            [width//2, height//10], 
            [width//10, height//3], 
            [width//20, height//2]
        ], np.int32)
        pts = pts.reshape((-1, 1, 2))
        cv2.polylines(img_copy, [pts], True, (0, 0, 255), 3)
        return img_copy
    elif operation == "add_text":
        img_copy = img.copy()
        height, width = img.shape[:2]
        text = 'OpenCV Drawing!'
        font = cv2.FONT_HERSHEY_SCRIPT_COMPLEX
        font_scale = max(width, height) / 800
        thickness = max(1, int(font_scale * 2))
        cv2.putText(img_copy, text, (width//10, height//2), font, font_scale, (40, 200, 0), thickness)
        return img_copy
    
    # Transformations
    elif operation == "translate_right":
        rows, cols = img.shape[:2]
        M = np.float32([[1, 0, 100], [0, 1, 0]])
        return cv2.warpAffine(img, M, (cols, rows))
    elif operation == "translate_down":
        rows, cols = img.shape[:2]
        M = np.float32([[1, 0, 0], [0, 1, 100]])
        return cv2.warpAffine(img, M, (cols, rows))
    elif operation == "rotate_45":
        rows, cols = img.shape[:2]
        center = (cols / 2, rows / 2)
        M = cv2.getRotationMatrix2D(center, 45, 1.0)
        return cv2.warpAffine(img, M, (cols, rows))
    elif operation == "rotate_90":
        return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    elif operation == "flip_horizontal":
        return cv2.flip(img, 1)
    elif operation == "flip_vertical":
        return cv2.flip(img, 0)
    
    # FIXED: Scaling and Resizing - DO NOT resize back to original
    elif operation == "resize_50":
        # Resize to 50% - keep the new smaller size
        height, width = img.shape[:2]
        new_width = int(width * 0.5)
        new_height = int(height * 0.5)
        return cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)
    elif operation == "resize_200":
        # Resize to 200% - keep the new larger size
        height, width = img.shape[:2]
        new_width = int(width * 2)
        new_height = int(height * 2)
        return cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
    elif operation == "crop_center":
        # Crop center 50% and keep cropped size (don't resize back)
        height, width = img.shape[:2]
        start_row, start_col = int(height * 0.25), int(width * 0.25)
        end_row, end_col = int(height * 0.75), int(width * 0.75)
        return img[start_row:end_row, start_col:end_col]
    
    # Arithmetic Operations
    elif operation == "brighten":
        return cv2.add(img, np.array([50.0]))
    elif operation == "darken":
        return cv2.subtract(img, np.array([50.0]))
    elif operation == "contrast":
        return cv2.multiply(img, np.array([1.5]))
    elif operation == "bitwise_not":
        return cv2.bitwise_not(img)
    
    # Blurring and Sharpening
    elif operation == "gaussian_blur":
        return cv2.GaussianBlur(img, (15, 15), 0)
    elif operation == "median_blur":
        return cv2.medianBlur(img, 15)
    elif operation == "bilateral_filter":
        return cv2.bilateralFilter(img, 9, 75, 75)
    elif operation == "sharpen":
        kernel = np.array([[-1,-1,-1], [-1, 9,-1], [-1,-1,-1]])
        return cv2.filter2D(img, -1, kernel)
    
    # Thresholding
    elif operation == "threshold_binary":
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        return cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
    elif operation == "threshold_otsu":
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
    elif operation == "adaptive_threshold":
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                       cv2.THRESH_BINARY, 11, 2)
        return cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
    
    # Morphological Operations
    elif operation == "dilate":
        kernel = np.ones((5,5), np.uint8)
        return cv2.dilate(img, kernel, iterations=1)
    elif operation == "erode":
        kernel = np.ones((5,5), np.uint8)
        return cv2.erode(img, kernel, iterations=1)
    elif operation == "morphological_gradient":
        kernel = np.ones((5,5), np.uint8)
        return cv2.morphologyEx(img, cv2.MORPH_GRADIENT, kernel)
    elif operation == "sobel_x":
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=5)
        sobelx = cv2.convertScaleAbs(sobelx)
        return cv2.cvtColor(sobelx, cv2.COLOR_GRAY2BGR)
    elif operation == "sobel_y":
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=5)
        sobely = cv2.convertScaleAbs(sobely)
        return cv2.cvtColor(sobely, cv2.COLOR_GRAY2BGR)
    elif operation == "laplacian":
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        laplacian = cv2.convertScaleAbs(laplacian)
        return cv2.cvtColor(laplacian, cv2.COLOR_GRAY2BGR)
    
    return img


def get_image_dimensions(img):
    """Get image dimensions using numpy."""
    height = int(img.shape[0])
    width = int(img.shape[1])
    depth = int(img.shape[2]) if len(img.shape) == 3 else 1
    
    return {
        "height": height,
        "width": width, 
        "depth": depth,
        "total_pixels": height * width
    }


# -------------------------
# Startup DB Connection Check
# -------------------------
@app.on_event("startup")
def test_db_connection():
    try:
        db = SessionLocal()
        db.execute(text("SELECT 1"))
        print("✅ Database connection successful!")
    except Exception as e:
        print("❌ Database connection failed:", e)
    finally:
        db.close()


# -------------------------
# Root Endpoint
# -------------------------
@app.get("/")
def read_root():
    return {
        "message": "Image Processing API",
        "version": "1.0.0",
        "docs": "/docs",
        "endpoints": {
            "batches": "/batches/",
            "images": "/images/",
            "process": "/process/"
        }
    }


# -------------------------
# Batch Endpoints
# -------------------------
@app.post("/batches/")
def create_batch(name: str = Form(...), filter: str = Form(None), db: Session = Depends(get_db)):
    new_batch = models.Batch(name=name, filter=filter)
    db.add(new_batch)
    db.commit()
    db.refresh(new_batch)
    return {"id": new_batch.id, "name": new_batch.name, "filter": new_batch.filter}


@app.get("/batches/")
def list_batches(db: Session = Depends(get_db)):
    return db.query(models.Batch).all()


@app.delete("/batches/{batch_id}")
def delete_batch(batch_id: int, db: Session = Depends(get_db)):
    batch = db.query(models.Batch).filter(models.Batch.id == batch_id).first()
    if not batch:
        raise HTTPException(status_code=404, detail="Batch not found")

    images = db.query(models.Image).filter(models.Image.batch_id == batch_id).all()
    for img in images:
        file_path = os.path.join(OUTPUT_DIR, img.filename)
        if os.path.exists(file_path):
            os.remove(file_path)
        db.delete(img)

    db.delete(batch)
    db.commit()
    return {"message": f"Batch {batch_id} and its images deleted successfully"}


@app.get("/batches/{batch_id}/images")
def get_batch_images(batch_id: int, db: Session = Depends(get_db)):
    images = db.query(models.Image).filter(models.Image.batch_id == batch_id).all()
    if not images:
        raise HTTPException(status_code=404, detail="No images found for this batch")
    return [
        {
            "id": img.id,
            "filename": img.filename,
            "filter": img.filter,
            "created_at": img.created_at,
        }
        for img in images
    ]


@app.get("/batches/{batch_id}/export")
def export_batch(batch_id: int, db: Session = Depends(get_db)):
    images = db.query(models.Image).filter(models.Image.batch_id == batch_id).all()
    if not images:
        raise HTTPException(status_code=404, detail="No images in this batch.")

    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w") as zip_file:
        for img in images:
            fname = img.filename or f"image_{img.id}.png"
            zip_file.writestr(fname, img.processed)
    zip_buffer.seek(0)
    return StreamingResponse(
        zip_buffer,
        media_type="application/x-zip-compressed",
        headers={"Content-Disposition": f"attachment; filename=batch_{batch_id}_images.zip"}
    )


# -------------------------
# Image Endpoints
# -------------------------

@app.post("/process/")
async def process_image(
    file: UploadFile,
    operation: str = Form(...),
    batch_id: int = Form(...),
    db: Session = Depends(get_db),
):
    batch = db.query(models.Batch).filter(models.Batch.id == batch_id).first()
    if not batch:
        raise HTTPException(status_code=400, detail="Batch ID does not exist")

    img = read_image(file)
    original_dimensions = get_image_dimensions(img)
    
    processed = process_with_filter(img, operation)
    if processed is None:
        raise HTTPException(status_code=500, detail="Image processing failed")
    
    processed_dimensions = get_image_dimensions(processed)

    filename = f"{uuid.uuid4().hex}.png"
    path = os.path.join(OUTPUT_DIR, filename)
    if not cv2.imwrite(path, processed):
        raise HTTPException(status_code=500, detail="Failed to save processed image")

    _, buffer = cv2.imencode(".png", processed)
    db_image = models.Image(
        batch_id=batch_id,
        filename=filename,
        filter=operation,
        processed=buffer.tobytes(),
    )
    db.add(db_image)
    db.commit()
    db.refresh(db_image)

    return StreamingResponse(io.BytesIO(buffer.tobytes()), media_type="image/png")


@app.post("/process-with-info/")
async def process_image_with_info(
    file: UploadFile,
    operation: str = Form(...),
    batch_id: int = Form(...),
    db: Session = Depends(get_db),
):
    """Process image and return both image data and dimensions info."""
    batch = db.query(models.Batch).filter(models.Batch.id == batch_id).first()
    if not batch:
        raise HTTPException(status_code=400, detail="Batch ID does not exist")

    img = read_image(file)
    original_dimensions = get_image_dimensions(img)
    
    processed = process_with_filter(img, operation)
    if processed is None:
        raise HTTPException(status_code=500, detail="Image processing failed")
    
    processed_dimensions = get_image_dimensions(processed)

    filename = f"{uuid.uuid4().hex}.png"
    path = os.path.join(OUTPUT_DIR, filename)
    if not cv2.imwrite(path, processed):
        raise HTTPException(status_code=500, detail="Failed to save processed image")

    _, buffer = cv2.imencode(".png", processed)
    db_image = models.Image(
        batch_id=batch_id,
        filename=filename,
        filter=operation,
        processed=buffer.tobytes(),
    )
    db.add(db_image)
    db.commit()
    db.refresh(db_image)

    import base64
    image_base64 = base64.b64encode(buffer.tobytes()).decode('utf-8')

    return {
        "success": True,
        "image_id": db_image.id,
        "filename": filename,
        "filter_applied": operation,
        "image_data": f"data:image/png;base64,{image_base64}",
        "original_dimensions": {
            "height": original_dimensions["height"],
            "width": original_dimensions["width"],
            "depth": original_dimensions["depth"],
            "total_pixels": original_dimensions["total_pixels"],
            "formatted": f"Height: {original_dimensions['height']} pixels, Width: {original_dimensions['width']} pixels, Depth: {original_dimensions['depth']} color components"
        },
        "processed_dimensions": {
            "height": processed_dimensions["height"],
            "width": processed_dimensions["width"], 
            "depth": processed_dimensions["depth"],
            "total_pixels": processed_dimensions["total_pixels"],
            "formatted": f"Height: {processed_dimensions['height']} pixels, Width: {processed_dimensions['width']} pixels, Depth: {processed_dimensions['depth']} color components"
        }
    }


@app.get("/images/{image_id}/download")
def download_image(image_id: int, format: str = "png", db: Session = Depends(get_db)):
    """Download processed image in specified format (png, jpg, jpeg)."""
    image = db.query(models.Image).filter(models.Image.id == image_id).first()
    if not image:
        raise HTTPException(status_code=404, detail="Image not found")
    
    nparr = np.frombuffer(image.processed, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    format = format.lower()
    if format in ["jpg", "jpeg"]:
        _, buffer = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 95])
        media_type = "image/jpeg"
        extension = "jpg"
    else:
        _, buffer = cv2.imencode(".png", img)
        media_type = "image/png"
        extension = "png"
    
    filename = f"processed_{image_id}.{extension}"
    
    return StreamingResponse(
        io.BytesIO(buffer.tobytes()),
        media_type=media_type,
        headers={"Content-Disposition": f"attachment; filename={filename}"}
    )


@app.delete("/images/{image_id}")
def delete_image(image_id: int, db: Session = Depends(get_db)):
    image = db.query(models.Image).filter(models.Image.id == image_id).first()
    if not image:
        raise HTTPException(status_code=404, detail="Image not found")

    file_path = os.path.join(OUTPUT_DIR, image.filename)
    if os.path.exists(file_path):
        os.remove(file_path)

    db.delete(image)
    db.commit()
    return {"message": f"Image {image_id} deleted successfully"}


@app.get("/images/")
def list_images(db: Session = Depends(get_db)):
    images = db.query(models.Image).all()
    return [
        {
            "id": img.id,
            "batch_id": img.batch_id,
            "filename": img.filename,
            "filter": img.filter,
            "created_at": img.created_at,
        }
        for img in images
    ]


@app.get("/images/{image_id}/file")
def get_image_file(image_id: int, db: Session = Depends(get_db)):
    image = db.query(models.Image).filter(models.Image.id == image_id).first()
    if not image:
        raise HTTPException(status_code=404, detail="Image not found")

    return StreamingResponse(io.BytesIO(image.processed), media_type="image/png")