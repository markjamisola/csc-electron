from fastapi import FastAPI, UploadFile, Form, Depends, HTTPException
from fastapi.responses import StreamingResponse
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
    return img


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

    # Delete all images linked to batch
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
            # Use filename or fallback
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
    # Validate batch
    batch = db.query(models.Batch).filter(models.Batch.id == batch_id).first()
    if not batch:
        raise HTTPException(status_code=400, detail="Batch ID does not exist")

    # Read + process
    img = read_image(file)
    processed = process_with_filter(img, operation)
    if processed is None:
        raise HTTPException(status_code=500, detail="Image processing failed")

    # Save to disk
    filename = f"{uuid.uuid4().hex}.png"
    path = os.path.join(OUTPUT_DIR, filename)
    if not cv2.imwrite(path, processed):
        raise HTTPException(status_code=500, detail="Failed to save processed image")

    # Save to DB
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

    # Return processed image stream
    return StreamingResponse(io.BytesIO(buffer.tobytes()), media_type="image/png")


@app.delete("/images/{image_id}")
def delete_image(image_id: int, db: Session = Depends(get_db)):
    image = db.query(models.Image).filter(models.Image.id == image_id).first()
    if not image:
        raise HTTPException(status_code=404, detail="Image not found")

    # Remove from disk
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
