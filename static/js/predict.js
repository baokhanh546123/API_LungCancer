const VALID_MODEL_LIST = ['model-handmade', 'model-resnet', 'model-mobinet'];
lucide.createIcons();

const elements = {
    sidebar: document.getElementById('sidebar'),
    mainContent: document.getElementById('main-content'),
    toggleBtn: document.getElementById('sidebar-toggle'),
    toggleIcon: document.getElementById('toggle-icon'),
    fileInput: document.getElementById('upload-img-hidden'),
    fileCount: document.getElementById('file-count'),
    modelSelect: document.getElementById('model-select'),
    modelForm: document.getElementById('model-form'),
    runModelBtn: document.getElementById('run-model-btn'),
    imagePreviewContainer: document.getElementById('original-image-container'),

    // Dialog elements
    dialog: document.getElementById('status-dialog'),
    statusIcon: document.getElementById('status-icon'),
    statusTitle: document.getElementById('status-title'),
    statusMessage: document.getElementById('status-message')
};

// Sidebar ---
let isSidebarOpen = true;
function toggleSidebar() {
    isSidebarOpen = !isSidebarOpen;
    const { sidebar, toggleBtn, toggleIcon, mainContent } = elements;

    sidebar.classList.toggle('-translate-x-full', !isSidebarOpen);
    toggleBtn.classList.toggle('left-64', isSidebarOpen);
    toggleBtn.classList.toggle('left-2', !isSidebarOpen);
    mainContent.classList.toggle('ml-64', isSidebarOpen);
    mainContent.classList.toggle('ml-0', !isSidebarOpen);

    toggleIcon.setAttribute('data-lucide', isSidebarOpen ? 'chevrons-left' : 'chevrons-right');
    lucide.createIcons();
}
elements.toggleBtn.addEventListener('click', toggleSidebar);

//Notification) ---
let hideTimeout;
function showStatus(type, title, msg, duration = 4000) {
    clearTimeout(hideTimeout);
    const { dialog, statusIcon, statusTitle, statusMessage } = elements;

    dialog.className = 'fixed top-5 right-5 z-50 p-4 rounded-lg shadow-xl transition-all duration-500 transform w-80';
    statusIcon.className = ''; 

    const configs = {
        success: { color: 'bg-green-100 text-green-800 border-green-400', icon: 'check-circle' },
        error: { color: 'bg-red-100 text-red-800 border-red-400', icon: 'x-octagon' },
        loading: { color: 'bg-blue-100 text-blue-800 border-blue-400', icon: 'loader-2 animate-spin' }
    };

    const config = configs[type] || configs.success;
    dialog.classList.add(...config.color.split(' '));
    statusTitle.textContent = title;
    statusMessage.textContent = msg;
    statusIcon.setAttribute('data-lucide', config.icon.split(' ')[0]);
    if(config.icon.includes('animate-spin')) statusIcon.classList.add('animate-spin');

    lucide.createIcons();
    dialog.classList.remove('opacity-0', 'translate-x-full');

    if (duration > 0 && type !== 'loading') {
        hideTimeout = setTimeout(() => {
            dialog.classList.add('opacity-0', 'translate-x-full');
        }, duration);
    }
}

//File & Preview---
let selectedFile = null;
elements.fileInput.addEventListener('change', (e) => {
    const file = e.target.files[0];
    if (!file) {
        elements.fileCount.textContent = 'No file chosen';
        selectedFile = null;
        return;
    }
    selectedFile = file;
    elements.fileCount.textContent = '1 file selected';        

    elements.imagePreviewContainer.innerHTML = `<p class="text-gray-500">Press "Run Model" to view analysis.</p>`;
    document.getElementById('grad-cam-image-container').innerHTML = `<p class="text-gray-500">Grad-CAM result will appear after processing.</p>`;
});

//Submit Form (API Call) ---
function setBtnLoading(isLoading) {
    elements.runModelBtn.disabled = isLoading;
    elements.runModelBtn.classList.toggle('opacity-50', isLoading);
    elements.runModelBtn.classList.toggle('cursor-not-allowed', isLoading);
}

elements.modelForm.addEventListener('submit', async (event) => {
    event.preventDefault();

    const selectedModel = elements.modelSelect.value;
    //const file = elements.fileInput.files[0];

    // Validation
    if (!selectedFile) return showStatus('error', 'Required', 'Please upload an image.');
    if (!VALID_MODEL_LIST.includes(selectedModel)) return showStatus('error', 'Invalid', 'Please select a valid model.');

    setBtnLoading(true);
    showStatus('loading', 'Processing...', `Running ${selectedModel}...`, 0);

    const formData = new FormData();
    formData.append('model', selectedModel);
    formData.append('upload-img', selectedFile);

    const updateText = (id , value , fallback = '-') => {
        const el = document.getElementById(id);
        if (el) el.textContent = value !== undefined && value !== null ? value : fallback;
    }

    const updateVisualStatus = (isPneumonia) => {
        const classCard = document.getElementById('class-card');
        const interpretationCard = document.getElementById('interpretation-card');
        const interpIcon = document.getElementById('interpretation-icon');

        if (isPneumonia) {
            classCard.className = "p-4 bg-red-100 border border-red-200 rounded-lg shadow-md animate-pulse";
            interpretationCard.className = "bg-red-50 border-l-4 border-red-500 p-5 rounded-r-lg shadow-md";
            interpIcon.setAttribute('data-lucide', 'alert-circle');
            interpIcon.className = "w-8 h-8 text-red-500";
        } else {
            classCard.className = "p-4 bg-green-100 border border-green-200 rounded-lg shadow-md";
            interpretationCard.className = "bg-green-50 border-l-4 border-green-500 p-5 rounded-r-lg shadow-md";
            interpIcon.setAttribute('data-lucide', 'check-circle');
            interpIcon.className = "w-8 h-8 text-green-500";
        }
        lucide.createIcons();
    }

    const updateInferenceUI = (data, originalFile) => {
        const res = data.result;
        const images = data.images;
        const isPneumonia = res.prediction_label === 'PNEUMONIA';

        updateVisualStatus(isPneumonia);
        updateText('info-predict', res.prediction_label);
        updateText('info-decision-score', res.decision_score?.toFixed(4));
        updateText('info-predict-confidence', res.prediction_confidence ? `${(res.prediction_confidence * 100).toFixed(2)}%` : '-');
        
        updateText('info-inference-time', res.run_time ? `${res.run_time.toFixed(4)}s` : '0.0000s');
        if (res.raw_probabilities) {
            const pneu = res.raw_probabilities['PNEUMONIA'] || res.raw_probabilities['pneumonia'] || 0;
            const norm = res.raw_probabilities['NORMAL'] || res.raw_probabilities['normal'] || 0;
            updateText('info-prob-pneumonia', pneu.toFixed(4));
            updateText('info-prob-normal', norm.toFixed(4));
        }
        updateText('info-interpretation', res.interpretation);
        
        const reader = new FileReader();
        reader.onload = (e) => {
            elements.imagePreviewContainer.innerHTML = `
                <img src="${e.target.result}" 
                    class="h-full w-auto object-cover block mx-auto shadow-md rounded-lg fade-in">
            `;
            elements.imagePreviewContainer.classList.remove('bg-gray-200');
            elements.imagePreviewContainer.classList.add('bg-white');
        };
        reader.readAsDataURL(originalFile);

        const gradCamContainer = document.getElementById('grad-cam-image-container');
        if (gradCamContainer) {
            if (images && images.gradcam) {
                // Chèn ảnh với hiệu ứng transition để hiện ra mượt mà
                gradCamContainer.innerHTML = `
                    <img src="${images.gradcam}" 
                        alt="Grad-CAM Visualization" 
                        class="h-full w-auto object-cover block mx-auto shadow-md rounded-lg transition-all duration-500 ease-in-out"
                        style="display: block;
                        onclick="window.open(this.src, '_blank')"
                    />
                `;
                gradCamContainer.classList.replace('bg-gray-200', 'bg-white');
            } else {
                gradCamContainer.innerHTML = `<p class="text-red-500 font-medium">Grad-CAM image not available</p>`;
            }
        }
    }

    elements.imagePreviewContainer.innerHTML = `
    <div class="flex flex-col items-center">
        <div class="animate-spin rounded-full h-10 w-10 border-b-2 border-red-600 mb-2"></div>
        <p class="text-red-600 text-sm animate-pulse">Processing Original...</p>
    </div>
    `;


    document.getElementById('grad-cam-image-container').innerHTML = `
    <div class="flex flex-col items-center">
        <div class="animate-spin rounded-full h-10 w-10 border-b-2 border-blue-600 mb-2"></div>
        <p class="text-blue-600 text-sm animate-pulse">Generating Heatmap...</p>
    </div>
    `;

    try {
        const response = await fetch(elements.modelForm.action, {
            method: 'POST',
            body: formData
        });

        if (!response.ok) throw new Error(await response.text());

        const data = await response.json(); // result structure depends on backend response

        if (data.status === "success"){
            updateInferenceUI(data , selectedFile); 
            showStatus('success', 'Analysis Complete', `Model ${data.model_used} finished.`);
        }else{
            throw new Error(data.error || 'Model processing failed');
        }
    } catch (err) {
        showStatus('error', 'Failed', err.message || 'Server error occurred.');
        console.error("Fetch Error:", err);
    } finally {
        setBtnLoading(false);
    }
});