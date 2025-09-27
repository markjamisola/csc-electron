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
    // Show original image
    const origImg = document.createElement("img");
    origImg.src = URL.createObjectURL(file);
    origImg.className =
      "max-w-xs max-h-60 mx-auto rounded-lg border-2 border-gray-700 shadow-sm hover:border-green-400 transition mb-4";
    originalContainer.appendChild(origImg);

    // Prepare request to backend
    const formData = new FormData();
    formData.append("file", file);
    formData.append("operation", operation);
    formData.append("batch_id", batchId);

    try {
      const response = await fetch("http://127.0.0.1:8000/process/", {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        throw new Error("Failed to process image");
      }

      const blob = await response.blob();

      // Convert blob → base64 so <img> can display it
      const reader = new FileReader();
      reader.onloadend = () => {
        const procImg = document.createElement("img");
        procImg.src = reader.result;
        procImg.className =
          "max-w-xs max-h-60 mx-auto rounded-lg border-2 border-gray-700 shadow-sm hover:border-green-400 transition mb-4";
        processedContainer.appendChild(procImg);
      };
      reader.readAsDataURL(blob);

      // Refresh saved images after upload
      fetchAndDisplayImages();
    } catch (err) {
      console.error("Processing error:", err);
      alert("Error processing image. Check console for details.");
    }
  }
}

async function fetchAndDisplayImages() {
  const response = await fetch("http://127.0.0.1:8000/images/");
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
      const imgUrl = `http://127.0.0.1:8000/images/${img.id}/file`;
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
