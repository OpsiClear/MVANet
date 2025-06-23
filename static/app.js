// Configuration
const API_BASE_URL = '/api';

// Store current processing state
let isProcessing = false;
let currentTaskId = null;
let latestImageModal;
let statusCheckInterval = null;
let lastLogTimestamp = null;

// Helper function to manage input field state
function setInputFieldState(isProcessing, folderPath = '') {
    const inputField = document.getElementById('inputFolder');
    
    if (isProcessing) {
        inputField.value = folderPath;
        inputField.style.backgroundColor = '#fff3cd'; // Light yellow background
        inputField.style.borderColor = '#ffeaa7'; // Yellow border
        inputField.disabled = true;
    } else {
        inputField.value = '';
        inputField.style.backgroundColor = '';
        inputField.style.borderColor = '';
        inputField.disabled = false;
    }
}

// Helper function to stop monitoring and reset state
function stopTaskMonitoring() {
    isProcessing = false;
    currentTaskId = null;
    updateUIForProcessing(false);
    setInputFieldState(false);
    
    if (statusCheckInterval) {
        clearInterval(statusCheckInterval);
        statusCheckInterval = null;
    }
}

// Helper function to format timestamp
function formatTimestamp() {
            const now = new Date();
    return `[${now.toLocaleTimeString()}]`;
}

// Console output functions
function addConsoleMessage(message, type = 'info') {
    const consoleOutput = document.getElementById('consoleOutput');
    const line = document.createElement('div');
    line.className = `console-line ${type}`;
    line.textContent = `${formatTimestamp()} ${message}`;
    
    consoleOutput.appendChild(line);
    
    // Auto-scroll to bottom
    consoleOutput.scrollTop = consoleOutput.scrollHeight;
}

function clearConsole() {
    const consoleOutput = document.getElementById('consoleOutput');
    consoleOutput.innerHTML = '<div class="console-line text-muted">Ready to process images...</div>';
    lastLogTimestamp = null;
}

// Fetch and display task logs
async function fetchTaskLogs(taskId) {
    try {
        const url = lastLogTimestamp 
            ? `${API_BASE_URL}/logs/${taskId}?since=${encodeURIComponent(lastLogTimestamp)}`
            : `${API_BASE_URL}/logs/${taskId}`;
        
        const response = await fetch(url);
        if (!response.ok) {
            throw new Error(`HTTP ${response.status}: ${response.statusText}`);
        }
        
        const data = await response.json();
        
        if (data.logs && data.logs.length > 0) {
            data.logs.forEach(log => {
                // Parse timestamp and format it
                const timestamp = new Date(log.timestamp);
                const timeStr = timestamp.toLocaleTimeString();
                
                // Map log levels to console types
                const type = log.level.toLowerCase() === 'error' ? 'error' :
                           log.level.toLowerCase() === 'warning' ? 'warning' : 'info';
                
                // Add the log message to console
                const consoleOutput = document.getElementById('consoleOutput');
                const line = document.createElement('div');
                line.className = `console-line ${type}`;
                line.textContent = `[${timeStr}] ${log.message}`;
                
                consoleOutput.appendChild(line);
                lastLogTimestamp = log.timestamp;
            });
            
            // Auto-scroll to bottom
            const consoleOutput = document.getElementById('consoleOutput');
            consoleOutput.scrollTop = consoleOutput.scrollHeight;
        }
        
    } catch (error) {
        console.error('Error fetching task logs:', error);
        // Don't show error in console for log fetching failures to avoid spam
    }
}

// System status functions
async function checkSystemStatus() {
    try {
        const response = await fetch(`${API_BASE_URL}/system/status`);
        if (!response.ok) {
            throw new Error(`HTTP ${response.status}: ${response.statusText}`);
        }
        
        const status = await response.json();
        
        if (status.is_processing && !isProcessing) {
            // Found a running task that we didn't know about (probably from before page refresh)
            isProcessing = true;
            currentTaskId = status.current_task_id;
            
            addConsoleMessage('='.repeat(50), 'info');
            addConsoleMessage('Detected running task from previous session', 'warning');
            addConsoleMessage(`Task ID: ${status.current_task_id}`, 'info');
            addConsoleMessage(`Processing: ${status.current_input_folder || 'Unknown folder'}`, 'info');
            addConsoleMessage('Fetching processing logs...', 'info');
            
            // Show current processing folder in the input field
            if (status.current_input_folder) {
                setInputFieldState(true, status.current_input_folder);
            }
            
            // Update UI to show processing state
            updateUIForProcessing(true);
            
            // Start monitoring the task
            startTaskMonitoring(status.current_task_id);
            
        } else if (!status.is_processing && isProcessing) {
            // Task completed
            stopTaskMonitoring();
        }
        
        return status;
        
    } catch (error) {
        console.error('Error checking system status:', error);
        return null;
    }
}

// Monitor a specific task
async function startTaskMonitoring(taskId) {
    if (statusCheckInterval) {
        clearInterval(statusCheckInterval);
    }
    
    // Fetch initial logs
    await fetchTaskLogs(taskId);
    
    statusCheckInterval = setInterval(async () => {
        try {
            // Fetch new logs first
            await fetchTaskLogs(taskId);
            
            // Then check task status
            const response = await fetch(`${API_BASE_URL}/status/${taskId}`);
            if (response.ok) {
                const taskStatus = await response.json();
                
                if (taskStatus.status === 'completed') {
                    // Fetch any final logs
                    await fetchTaskLogs(taskId);
                    
                    addConsoleMessage('='.repeat(50), 'success');
                    addConsoleMessage('Task completed successfully!', 'success');
                    addConsoleMessage(`Output folder: ${taskStatus.output_folder}`, 'success');
                    
                    // Stop monitoring
                    stopTaskMonitoring();
                    
                } else if (taskStatus.status === 'failed') {
                    // Fetch any final logs
                    await fetchTaskLogs(taskId);
                    
                    addConsoleMessage('='.repeat(50), 'error');
                    addConsoleMessage(`Task failed: ${taskStatus.error_message}`, 'error');
                    addConsoleMessage('Processing failed!', 'error');
                    
                    // Stop monitoring
                    stopTaskMonitoring();
                }
                // If still processing, logs are fetched above - no need for generic message
                
            } else if (response.status === 404) {
                // Task not found, probably completed and cleaned up
                addConsoleMessage('Task monitoring ended (task not found)', 'info');
                stopTaskMonitoring();
            }
        } catch (error) {
            console.error('Error monitoring task:', error);
        }
    }, 2000); // Check every 2 seconds
}

// Update UI based on processing state
function updateUIForProcessing(processing) {
    const submitButton = document.querySelector('#jobForm button[type="submit"]');
    
    if (processing) {
        submitButton.innerHTML = '<span class="spinner-border spinner-border-sm me-2"></span>Processing...';
        submitButton.disabled = true;
    } else {
        submitButton.innerHTML = 'Submit Job';
        submitButton.disabled = false;
    }
}


    


// Show latest processed image
async function showLatestImage() {
    try {
        addConsoleMessage('Loading latest processed image...', 'info');
        
        const response = await fetch(`${API_BASE_URL}/latest-image`);
        if (!response.ok) {
            if (response.status === 404) {
                throw new Error('No processed images found');
            }
            throw new Error(`HTTP ${response.status}: ${response.statusText}`);
        }
        
        const imageData = await response.json();
        
        addConsoleMessage(`Found latest image: ${imageData.image_name}`, 'success');
        
        
        
        // Create image content
        const latestImageContent = document.getElementById('latestImageContent');
        const imageHtml = `
            <div class="mb-3">
                <img src="${imageData.image_url}" 
                     class="img-fluid rounded shadow" 
                     alt="Latest processed image"
                     style="max-height: 500px; cursor: pointer;"
                     onclick="window.open('${imageData.image_url}', '_blank')"
                     onload="this.style.opacity='1'"
                     onerror="this.src='data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMjAwIiBoZWlnaHQ9IjIwMCIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj48cmVjdCB3aWR0aD0iMjAwIiBoZWlnaHQ9IjIwMCIgZmlsbD0iI2Y4ZjlmYSIvPjx0ZXh0IHg9IjUwJSIgeT0iNTAlIiBmb250LWZhbWlseT0iQXJpYWwiIGZvbnQtc2l6ZT0iMTRweCIgZmlsbD0iIzY5NzY4OSIgdGV4dC1hbmNob3I9Im1pZGRsZSIgZHk9Ii4zZW0iPkZhaWxlZCB0byBsb2FkPC90ZXh0Pjwvc3ZnPg=='"
                     style="opacity: 0; transition: opacity 0.3s;">
            </div>
            <div class="text-center">
                <code>${imageData.output_folder}\\${imageData.image_name}</code>
            </div>
        `;
        
        latestImageContent.innerHTML = imageHtml;
        latestImageModal.show();
        
    } catch (error) {
        addConsoleMessage(`Error loading latest image: ${error.message}`, 'error');
        console.error('Error loading latest image:', error);
    }
}

// Initialize the app
document.addEventListener('DOMContentLoaded', async function() {
    // Initialize modal
    latestImageModal = new bootstrap.Modal(document.getElementById('latestImageModal'));
    
    // Handle form submission
    document.getElementById('jobForm').addEventListener('submit', submitJob);
    
    // Handle clear console button
    document.getElementById('clearConsole').addEventListener('click', clearConsole);
    
    // Handle show latest image button
    document.getElementById('showLatestImage').addEventListener('click', showLatestImage);
    
    addConsoleMessage('Application initialized', 'info');
    
    // Check for any running tasks on page load
    addConsoleMessage('Checking for running tasks...', 'info');
    await checkSystemStatus();
    
    // If no tasks are running, show ready message
    if (!isProcessing) {
        addConsoleMessage('No running tasks detected', 'info');
    }
});

// Submit a new job
async function submitJob(event) {
    event.preventDefault();
    
    if (isProcessing) {
        addConsoleMessage('Another job is already processing. Please wait...', 'warning');
        return;
    }
    
    const inputFolder = document.getElementById('inputFolder').value.trim();
    const useTta = document.getElementById('useTta').checked;
    
    if (!inputFolder) {
        addConsoleMessage('Please enter an input folder path', 'error');
        return;
    }
    
    const request = {
        input_folder: inputFolder,
        use_tta: useTta
    };
    
    try {
        // Immediately freeze the input field and update UI
        isProcessing = true;
        setInputFieldState(true, inputFolder);
        updateUIForProcessing(true);
        
        // Add console messages
        addConsoleMessage('='.repeat(50), 'info');
        addConsoleMessage(`Starting image segmentation job`, 'info');
        addConsoleMessage(`Input folder: ${inputFolder}`, 'info');
        addConsoleMessage(`Test-Time Augmentation: ${useTta ? 'Enabled' : 'Disabled'}`, 'info');
        addConsoleMessage('Sending request to server...', 'info');
        
        const response = await fetch(`${API_BASE_URL}/process`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(request)
        });
        
        if (!response.ok) {
            const errorData = await response.json();
            if (response.status === 409) {
                throw new Error('Another task is already being processed. Please wait for it to complete.');
            }
            throw new Error(errorData.detail || `Server error: ${response.status}`);
        }
        
        const result = await response.json();
        
        if (result.status === 'processing') {
            // Task submitted successfully for background processing
            currentTaskId = result.request_id;
            
            addConsoleMessage('Task submitted successfully!', 'success');
            addConsoleMessage(`Task ID: ${result.request_id}`, 'info');
            addConsoleMessage('Processing started in background...', 'info');
            addConsoleMessage('You can now safely refresh the page', 'info');
            addConsoleMessage('Fetching real-time processing logs...', 'info');
            
            // Reset TTA checkbox only
            document.getElementById('useTta').checked = true;
            
            // Start monitoring the task
            startTaskMonitoring(result.request_id);
            
        } else if (result.status === 'failed') {
            throw new Error(result.error_message || 'Unknown error occurred');
        } else {
            // Immediate completion (unlikely but handle it)
            addConsoleMessage('Task completed immediately!', 'success');
            addConsoleMessage(`Output folder: ${result.output_folder}`, 'success');
        }
        
    } catch (error) {
        addConsoleMessage(`Error: ${error.message}`, 'error');
        addConsoleMessage('Failed to submit job!', 'error');
        console.error('Error submitting job:', error);
        
        // Reset state on error (restore input field)
        stopTaskMonitoring();
    }
}