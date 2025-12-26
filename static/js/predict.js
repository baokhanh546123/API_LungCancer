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
elements.fileInput.addEventListener('change', (e) => {
    const file = e.target.files[0];
    if (!file) {
        elements.fileCount.textContent = 'No file chosen';
        return;
    }
    elements.fileCount.textContent = '1 file selected';

    const reader = new FileReader();
    reader.onload = (event) => {
        elements.imagePreviewContainer.innerHTML = `<img src="${event.target.result}" class="max-w-full max-h-full object-contain">`;
    };
    reader.readAsDataURL(file);
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
    const file = elements.fileInput.files[0];

    // Validation
    if (!file) return showStatus('error', 'Required', 'Please upload an image.');
    if (!VALID_MODEL_LIST.includes(selectedModel)) return showStatus('error', 'Invalid', 'Please select a valid model.');

    setBtnLoading(true);
    showStatus('loading', 'Processing...', `Running ${selectedModel}...`, 0);

    const formData = new FormData();
    formData.append('model', selectedModel);
    formData.append('upload-img', file);

    try {
        const response = await fetch(elements.modelForm.action, {
            method: 'POST',
            body: formData
        });

        if (!response.ok) throw new Error(await response.text());

        const result = await response.json();
        showStatus('success', 'Success!', `Model: ${result.model_used}`);
        console.log("Response:", result);
    } catch (err) {
        showStatus('error', 'Failed', 'Server error or connection lost.');
        console.error(err);
    } finally {
        setBtnLoading(false);
    }
});