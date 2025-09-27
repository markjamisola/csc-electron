async function processImage() {
  const fileInput = document.getElementById("fileInput");
  const operation = document.getElementById("operation").value;
  const batchId = getSelectedBatchId();
  const originalContainer = document.getElementById("original");
  const processedContainer = document.getElementById("processed");

  const files = fileInput.files;
  if (!files.length) {
    alert("Upload at least one image!");
    return;
  }
  if (!batchId) {
    alert("Please create/select a batch first!");
    return;
  }

  // Clear old results
  originalContainer.innerHTML = "";
  processedContainer.innerHTML = "";

  for (const file of files) {
    // Show original image with dimensions
    const origImg = document.createElement("img");
    origImg.src = URL.createObjectURL(file);
    origImg.className =
      "max-w-xs max-h-60 mx-auto rounded-lg border-2 border-gray-700 shadow-sm hover:border-green-400 transition mb-2";
    
    // Add onload event to get original image dimensions
    origImg.onload = function() {
      const origDimensionsDiv = document.createElement("div");
      origDimensionsDiv.className = "text-xs text-gray-300 mb-4 p-2 bg-gray-800 rounded";
      origDimensionsDiv.innerHTML = `
        <h4 class="text-blue-400 font-semibold mb-1">Original Image:</h4>
        <div>Height: ${this.naturalHeight} pixels</div>
        <div>Width: ${this.naturalWidth} pixels</div>
        <div>File Size: ${(file.size / 1024).toFixed(1)} KB</div>
      `;
      originalContainer.appendChild(origDimensionsDiv);
    };
    
    originalContainer.appendChild(origImg);

    // Prepare request to backend
    const formData = new FormData();
    formData.append("file", file);
    formData.append("operation", operation);
    formData.append("batch_id", batchId);

    try {
      const response = await fetch("http://127.0.0.1:8001/process-with-info/", {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        throw new Error("Failed to process image");
      }

      const result = await response.json();

      // Display processed image
      const procImg = document.createElement("img");
      procImg.src = result.image_data;
      procImg.className =
        "max-w-xs max-h-60 mx-auto rounded-lg border-2 border-gray-700 shadow-sm hover:border-green-400 transition mb-4";
      processedContainer.appendChild(procImg);

      // Display image dimensions info
      const dimensionsDiv = document.createElement("div");
      dimensionsDiv.className = "text-xs text-gray-300 mt-2 p-2 bg-gray-800 rounded";
      dimensionsDiv.innerHTML = `
        <h4 class="text-green-400 font-semibold mb-1">Image Dimensions:</h4>
        <div class="mb-1"><strong>Original:</strong> ${result.original_dimensions.formatted}</div>
        <div><strong>Processed:</strong> ${result.processed_dimensions.formatted}</div>
        <div class="text-gray-400 text-xs mt-1">Filter Applied: ${result.filter_applied}</div>
      `;
      processedContainer.appendChild(dimensionsDiv);

      // Refresh saved images after upload
      fetchAndDisplayImages();
    } catch (err) {
      console.error("Processing error:", err);
      alert("Error processing image. Check console for details.");
    }
  }
}

async function fetchAndDisplayImages() {
  const response = await fetch("http://127.0.0.1:8001/images/");
  const images = await response.json();
  const container = document.getElementById("batchContainer");
  if (!container) return; // only exists on savedbatches.html

  container.innerHTML = "<h2 class='text-xl text-green-400 mb-4'>Saved Batches</h2>";

  if (images.length === 0) {
    container.innerHTML += "<p class='text-gray-400'>No images found.</p>";
    return;
  }

  // Group by batch_id
  const grouped = images.reduce((acc, img) => {
    if (!acc[img.batch_id]) acc[img.batch_id] = [];
    acc[img.batch_id].push(img);
    return acc;
  }, {});

  Object.keys(grouped).forEach((batchId) => {
    container.innerHTML += `
      <div class="mb-8 p-4 bg-gray-800 rounded-lg">
        <h3 class="text-lg font-semibold text-green-400 mb-3">Batch #${batchId}</h3>
        <div class="grid grid-cols-2 md:grid-cols-4 gap-4" id="batch-${batchId}"></div>
      </div>
    `;

    const batchDiv = document.getElementById(`batch-${batchId}`);
    grouped[batchId].forEach((img) => {
      const imgUrl = `http://127.0.0.1:8001/images/${img.id}/file`;
      batchDiv.innerHTML += `
        <div class="bg-gray-900 p-2 rounded-lg shadow hover:scale-105 transition">
          <img src="${imgUrl}" class="max-w-[150px] max-h-[150px] mx-auto rounded-lg border border-gray-700" />
        </div>
      `;
    });
  });
}

// Load saved images on startup (only runs on savedbatches.html)
window.addEventListener("DOMContentLoaded", fetchAndDisplayImages);
