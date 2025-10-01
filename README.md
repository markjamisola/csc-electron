# 📷 Image Processing Studio

A modern desktop application for image processing using **OpenCV**, **FastAPI**, and **Electron**. Built for CSC 126 Computer Vision coursework.

## 👥 Development Team

- **Mark Jamisola** → `mark-dev` branch
- **Albert Agbo** → `albert-dev` branch  
- **Ushyne Esclamado** → `ushyne-dev` branch

**Course**: CSC 126 - CJ1 - GRAPHICS AND VISUAL COMPUTING
**Semester**: 1st Semester, 4th Year  
**Academic Year**: 2025

## ✨ Features

### **🎨 Image Processing Operations**
- **HSV Color Space Operations**: Blue, Green, Red channel isolation and full HSV conversion
- **BGR Channel Operations**: Grayscale conversion, channel isolation, and enhancement (+100 boost)
- **Basic Filters**: Grayscale, Canny edge detection, and Gaussian blur
- **OpenCV Drawing Tools**: Add lines, rectangles, circles, polygons, and text to images

### **📊 Advanced Image Analysis**
- **Detailed Dimension Display**: Shows original and processed image dimensions
- **File Size Information**: Displays file size for uploaded images
- **Real-time Processing**: Instant filter application with dimension analysis
- **Image Comparison**: Side-by-side original vs processed image display

### **📁 Batch Management & Export**
- **Batch Creation**: Create and organize multiple images into batches
- **Batch Export**: Export entire batches as ZIP files for easy download
- **Saved Batches View**: Dedicated interface to manage and view all saved batches
- **Image Gallery**: Grid-based image display with batch organization

### **💻 User Interface**
- **Modern UI**: Clean, responsive desktop interface built with Electron and Tailwind CSS
- **Enhanced Image Display**: Improved image viewing with dimension overlays
- **Updated Dependencies**: Latest Axios (1.12.2) and Electron (38.1.2) versions
- **Responsive Design**: Works seamlessly across different screen sizes

## 🛠 Technology Stack

- **Frontend**: Electron, HTML5, Tailwind CSS, JavaScript
- **Backend**: Python FastAPI, OpenCV (cv2), SQLAlchemy
- **Database**: PostgreSQL (Docker container)
- **Image Processing**: OpenCV Python library
- **Architecture**: REST API with desktop client

---

## 📋 Prerequisites

Before you start, make sure you have these installed on your computer:

### Required Software:

1. **Python 3.8+** - [Download Python](https://www.python.org/downloads/)
   - ⚠️ **Important**: During installation, check "Add Python to PATH"
   - Verify: Open Command Prompt/Terminal and run: `python --version`

2. **Node.js 16+** - [Download Node.js](https://nodejs.org/)
   - This includes npm (Node Package Manager)
   - Verify: Open Command Prompt/Terminal and run: `node --version` and `npm --version`

3. **Docker Desktop** - [Download Docker](https://www.docker.com/products/docker-desktop/)
   - Required for PostgreSQL database
   - Verify: Open Command Prompt/Terminal and run: `docker --version`

4. **Git** - [Download Git](https://git-scm.com/downloads/)
   - For cloning the repository
   - Verify: Open Command Prompt/Terminal and run: `git --version`

---

## 🚀 Installation & Setup

### ⚡ Quick Setup Checklist

Before you start, make sure you have:
- [ ] Python 3.8+ installed and added to PATH
- [ ] Node.js 16+ installed  
- [ ] Docker Desktop installed and running
- [ ] Git installed
- [ ] A new terminal/command prompt open

### 📝 Setup Overview

1. **Clone repository** and start database
2. **Set up Python virtual environment** (important!)
3. **Install Python dependencies** and start backend server
4. **Install Node.js dependencies** and start Electron app
5. **Test everything** by creating a batch and processing images

Follow these steps **exactly** to set up the application:

### Step 1: Clone the Repository

```bash
git clone https://github.com/markjamisola/csc-electron.git
cd csc-electron
```

### Step 2: Set Up the Database (PostgreSQL)

1. **Start Docker Desktop** on your computer
2. **Start the PostgreSQL container**:
   ```bash
   docker-compose up -d
   ```
3. **Verify database is running**:
   ```bash
   docker ps
   ```
   You should see a container with PostgreSQL running on port 5432.

### Step 3: Set Up the Backend (Python FastAPI)

1. **Navigate to backend directory**:
   ```bash
   cd backend
   ```

2. **Create a Python virtual environment** (recommended):
   ```bash
   # On Windows:
   python -m venv venv
   
   # On Mac/Linux:
   python3 -m venv venv
   ```

3. **Activate the virtual environment**:
   ```bash
   # On Windows:
   venv\Scripts\activate
   
   # On Mac/Linux:
   source venv/bin/activate
   ```
   
   ✅ **You should see `(venv)` at the beginning of your command prompt**

4. **Upgrade pip** (important for avoiding installation issues):
   ```bash
   python -m pip install --upgrade pip
   ```

5. **Install Python dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
   
   ⚠️ **If you encounter errors, try these solutions:**
   ```bash
   # For Windows users with long path issues:
   pip install --user -r requirements.txt
   
   # For permission issues:
   pip install -r requirements.txt --user
   
   # For specific package issues:
   pip install --upgrade setuptools wheel
   pip install -r requirements.txt
   ```

6. **Start the backend server**:
   ```bash
   python -m uvicorn main:app --reload --port 8001
   ```
   
   ✅ **Success indicators**:
   - You should see: `✅ Database connection successful!`
   - Server running on: `http://127.0.0.1:8001`
   - No error messages in the terminal

   ⚠️ **Keep this terminal open** - the backend server needs to stay running!

### Step 4: Set Up the Frontend (Electron)

**Open a NEW terminal/command prompt** (keep the backend running in the first one).

⚠️ **Important**: If you're using the same terminal later, you'll need to:
1. Navigate to the backend directory: `cd backend`
2. Activate the virtual environment: `venv\Scripts\activate` (Windows) or `source venv/bin/activate` (Mac/Linux)
3. Start the backend server: `python -m uvicorn main:app --reload --port 8001`

**For the frontend (in the NEW terminal):**

1. **Navigate to frontend directory**:
   ```bash
   cd frontend
   ```

2. **Install Node.js dependencies**:
   ```bash
   npm install
   ```
   
   ⚠️ **If you get errors, try:**
   ```bash
   # Clear npm cache
   npm cache clean --force
   
   # Delete node_modules and reinstall
   rm -rf node_modules
   npm install
   
   # On Windows, if you get permission errors:
   npm install --no-optional
   ```

3. **Start the Electron application**:
   ```bash
   npx electron .
   ```
   
   ✅ **Success**: The Image Processing Studio desktop app should open!

---

## ⚠️ Important Notes

### Every Time You Want to Run the Application:

**Terminal 1 (Backend):**
```bash
cd backend
venv\Scripts\activate          # Windows
# OR: source venv/bin/activate   # Mac/Linux
python -m uvicorn main:app --reload --port 8001
```

**Terminal 2 (Frontend):**
```bash
cd frontend
npx electron .
```

### Before You Start:
1. ✅ **Docker Desktop must be running**
2. ✅ **PostgreSQL container must be started**: `docker-compose up -d`
3. ✅ **Virtual environment must be activated** (you'll see `(venv)` in the prompt)
4. ✅ **Both terminals must be kept open** while using the application

---

## 🎯 How to Use the Application

### 1. Create a Batch
- Click **"New Batch"** button
- Enter a name for your batch (e.g., "My Images")
- Click **"Create Batch"**

### 2. Upload Images
- Click on the **upload area** or drag & drop images
- Supported formats: JPG, PNG, GIF, BMP
- Multiple images can be uploaded at once

### 3. Process Images
- Click on any uploaded image to select it
- Choose a filter from the **Processing Options** dropdown:
  
  **HSV Color Operations:**
  - HSV Color Space
  - HSV - Blue Channel Only
  - HSV - Green Channel Only
  - HSV - Red Channel Only
  
  **BGR Channel Operations:**
  - Blue/Green/Red Channel (Grayscale)
  - Blue/Green/Red Channel Isolation
  - Blue/Green/Red Channel Enhancement
  
  **Drawing & Shapes:**
  - Add Diagonal Line
  - Add Rectangle
  - Add Circle
  - Add Polygon
  - Add Text

- Click **"Apply Filter"** to process the image
- View results with original vs processed dimensions

### 4. Image Information
- Click **"Show Dimensions"** to see detailed image properties
- View height, width, depth, and total pixels

---

## 🔧 Troubleshooting

### Common Issues & Solutions:

#### ❌ "Database connection failed"
**Solution:**
1. Make sure Docker Desktop is running
2. Start the database: `docker-compose up -d`
3. Wait 10-15 seconds, then restart the backend server

#### ❌ "Module not found" errors
**Solution:**
1. Make sure your virtual environment is activated (you should see `(venv)` in your prompt)
2. If not activated, run: `venv\Scripts\activate` (Windows) or `source venv/bin/activate` (Mac/Linux)
3. Make sure you're in the correct directory (`backend/`)
4. Reinstall dependencies: `pip install -r requirements.txt`
5. Try using: `python -m pip install [package-name]`

#### ❌ "Virtual environment issues"
**Solution:**
1. Delete the venv folder: `rm -rf venv` (Mac/Linux) or `rmdir /s venv` (Windows)
2. Recreate virtual environment: `python -m venv venv`
3. Activate it: `venv\Scripts\activate` (Windows) or `source venv/bin/activate` (Mac/Linux)
4. Reinstall dependencies: `pip install -r requirements.txt`

#### ❌ "Permission denied" or "Access denied"
**Solution:**
1. Run terminal as Administrator (Windows) or use `sudo` (Mac/Linux)
2. Or install with user flag: `pip install -r requirements.txt --user`
3. Make sure antivirus is not blocking the installation

#### ❌ "Port already in use"
**Solution:**
1. Stop any existing servers using the ports
2. On Windows: `netstat -ano | findstr :8001` then `taskkill /PID [PID_NUMBER] /F`
3. On Mac/Linux: `lsof -ti:8001 | xargs kill -9`

#### ❌ Frontend won't start
**Solution:**
1. Make sure Node.js is installed: `node --version`
2. Clear npm cache: `npm cache clean --force`
3. Delete `node_modules` and reinstall: 
   ```bash
   rm -rf node_modules
   npm install
   ```

#### ❌ Images not processing
**Solution:**
1. Ensure backend server is running on port 8001
2. Check if batch is created properly
3. Try uploading images in supported formats (JPG, PNG, GIF, BMP)

---

## 📂 Project Structure

```
csc-electron/
├── backend/                 # Python FastAPI server
│   ├── main.py             # Main server application
│   ├── models.py           # Database models
│   ├── database.py         # Database configuration
│   ├── requirements.txt    # Python dependencies
│   └── outputs/            # Processed images storage
├── frontend/               # Electron desktop app
│   ├── index.html          # Main application UI
│   ├── main.js             # Electron main process
│   ├── renderer.js         # Electron renderer process
│   ├── package.json        # Node.js dependencies
│   └── savedbatches.html   # Saved batches view
├── docker-compose.yaml     # PostgreSQL database setup
└── README.md              # This file
```

---

## 🔗 API Endpoints

The backend provides these REST API endpoints:

- `GET /` - Health check
- `POST /batches/` - Create new batch
- `GET /batches/` - List all batches
- `POST /process-with-info/` - Process image with details
- `GET /images/` - List all processed images
- `DELETE /images/{id}` - Delete processed image
- `GET /images/{id}/file` - Download processed image

---

## 📞 Need Help?

If you encounter any issues:

1. **Check this README** first - most common issues are covered
2. **Verify all prerequisites** are installed correctly
3. **Make sure all services are running**:
   - Docker Desktop
   - PostgreSQL container (`docker ps`)
   - Backend server (port 8001)
4. **Check the console/terminal** for error messages

---

## 🎉 You're Ready!

Once everything is set up, you should have:
- ✅ PostgreSQL database running in Docker
- ✅ Python FastAPI backend server on port 8001
- ✅ Electron desktop application running
- ✅ All image processing features working

**Happy image processing!** 📸✨
