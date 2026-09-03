(() => {
  'use strict';
  const uploadScreen = document.querySelector('#uploadScreen');
  const processingScreen = document.querySelector('#processingScreen');
  const resultsScreen = document.querySelector('#resultsScreen');
  const fatalErrorScreen = document.querySelector('#fatalErrorScreen');
  const dropzone = document.querySelector('#dropzone');
  const fileInput = document.querySelector('#fileInput');
  const preview = document.querySelector('#videoPreview');
  const measureButton = document.querySelector('#measureBtn');
  const errorBanner = document.querySelector('#uploadError');
  const errorText = document.querySelector('#uploadErrorText');
  let selectedFile = null;
  let objectUrl = null;
  let messageTimer = null;

  const show = screen => [uploadScreen, processingScreen, resultsScreen, fatalErrorScreen].forEach(item => item.classList.toggle('hidden', item !== screen));
  const showError = text => { errorText.textContent = text; errorBanner.classList.remove('hidden'); };
  const clearError = () => errorBanner.classList.add('hidden');
  const accepted = file => file && /\.(mp4|mov|avi|mkv)$/i.test(file.name);

  function setFile(file) {
    clearError();
    if (!accepted(file)) return showError('Please choose an MP4, MOV, AVI, or MKV video.');
    selectedFile = file;
    if (objectUrl) URL.revokeObjectURL(objectUrl);
    objectUrl = URL.createObjectURL(file);
    preview.src = objectUrl;
    document.querySelector('#fileName').textContent = file.name;
    preview.onloadedmetadata = () => { document.querySelector('#fileDuration').textContent = `· ${Math.floor(preview.duration / 60)}:${String(Math.floor(preview.duration % 60)).padStart(2, '0')}`; };
    document.querySelector('#dropzoneEmpty').classList.add('hidden');
    document.querySelector('#dropzonePreview').classList.remove('hidden');
    measureButton.disabled = false;
  }
  function resetUpload() {
    selectedFile = null; fileInput.value = ''; preview.removeAttribute('src');
    if (objectUrl) URL.revokeObjectURL(objectUrl); objectUrl = null;
    document.querySelector('#dropzoneEmpty').classList.remove('hidden');
    document.querySelector('#dropzonePreview').classList.add('hidden');
    measureButton.disabled = true; clearError();
  }

  dropzone.addEventListener('click', event => { if (!event.target.closest('#removeVideoBtn')) fileInput.click(); });
  dropzone.addEventListener('keydown', event => { if (event.key === 'Enter' || event.key === ' ') { event.preventDefault(); fileInput.click(); } });
  ['dragenter', 'dragover'].forEach(name => dropzone.addEventListener(name, event => { event.preventDefault(); dropzone.classList.add('drag-over'); }));
  ['dragleave', 'drop'].forEach(name => dropzone.addEventListener(name, event => { event.preventDefault(); dropzone.classList.remove('drag-over'); }));
  dropzone.addEventListener('drop', event => setFile(event.dataTransfer.files[0]));
  fileInput.addEventListener('change', event => setFile(event.target.files[0]));
  document.querySelector('#removeVideoBtn').addEventListener('click', event => { event.stopPropagation(); resetUpload(); });
  document.querySelector('#paperType').addEventListener('change', event => document.querySelector('#customPaperFields').classList.toggle('hidden', event.target.value !== 'CUSTOM'));

  function processingMessages() {
    const messages = ['Validating the video…', 'Finding the paper boundary…', 'Correcting perspective…', 'Tracking visible pen motion…', 'Calculating the ink estimate…'];
    let index = 0; document.querySelector('#processingMessage').textContent = messages[index];
    messageTimer = setInterval(() => { index = (index + 1) % messages.length; document.querySelector('#processingMessage').textContent = messages[index]; }, 1400);
  }
  function stopMessages() { clearInterval(messageTimer); messageTimer = null; }
  function formatInk(ml) { return ml < 0.001 ? `${(ml * 1000).toFixed(3)} µL` : `${ml.toFixed(6)} mL`; }

  function showResults(data) {
    const distance = data.scribble_length_cm;
    document.querySelector('#resultDistance').textContent = distance.toFixed(1);
    document.querySelector('#resultDistanceUnit').textContent = 'cm';
    const inkParts = formatInk(data.estimated_ink_ml).split(' ');
    document.querySelector('#resultInk').textContent = inkParts[0];
    document.querySelector('#resultInkUnit').textContent = inkParts[1];
    document.querySelector('#resultJoke').textContent = `That is ${distance.toFixed(1)} cm of determined scribbling.`;
    const values = [
      ['Paper', `${data.paper.type} · ${data.paper.width_mm} × ${data.paper.height_mm} mm`],
      ['Processing time', `${data.processing_seconds} seconds`],
      ['Tracking confidence', `${Math.round(data.confidence * 100)}%`],
      ['Accepted points', data.tracking.accepted_track_points],
      ['Ink rate', `${data.ink_rate_ml_per_m} mL/m`],
    ];
    const details = document.querySelector('#resultDetails');
    details.replaceChildren(...values.map(([name, value]) => { const row = document.createElement('div'); const label = document.createElement('dt'); const content = document.createElement('dd'); label.textContent = name; content.textContent = value; row.append(label, content); return row; }));
    show(resultsScreen);
  }

  measureButton.addEventListener('click', async () => {
    if (!selectedFile) return;
    const paperType = document.querySelector('#paperType').value;
    if (paperType === 'CUSTOM' && (!document.querySelector('#widthMm').value || !document.querySelector('#heightMm').value)) return showError('Enter width and height for custom paper.');
    show(processingScreen); processingMessages();
    const formData = new FormData();
    formData.append('video', selectedFile);
    formData.append('paper_type', paperType);
    formData.append('width_mm', document.querySelector('#widthMm').value);
    formData.append('height_mm', document.querySelector('#heightMm').value);
    formData.append('pen_brand', document.querySelector('#penBrand').value);
    formData.append('tip_size_mm', document.querySelector('#tipSizeMm').value);
    formData.append('pen_color', 'Blue');
    formData.append('ink_rate_ml_per_m', document.querySelector('#inkRate').value);
    try {
      const response = await fetch('/api/analyze', { method: 'POST', body: formData });
      const data = await response.json();
      if (!response.ok || !data.success) throw new Error(data.error || 'Analysis failed.');
      showResults(data);
    } catch (error) { document.querySelector('#fatalErrorText').textContent = error.message || 'Something unexpected happened while measuring your scribble.'; show(fatalErrorScreen); }
    finally { stopMessages(); }
  });
  document.querySelector('#resetBtn').addEventListener('click', () => { resetUpload(); show(uploadScreen); });
  document.querySelector('#tryAgainBtn').addEventListener('click', () => { show(uploadScreen); });
})();
