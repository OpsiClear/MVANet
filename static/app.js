// Configuration
const API_BASE_URL = '/api'; // API base path
const REFRESH_INTERVAL = 5000; // 5 seconds

// Store tasks
let tasks = [];
let taskModal;
let currentLogLevel = null;

// Deprecated but kept for backward compatibility
function loadTasksFromLocalStorage() {
    try {
        const tasksJson = localStorage.getItem('tasks');
        return tasksJson ? JSON.parse(tasksJson) : [];
    } catch (error) {
        console.error('Error loading tasks from localStorage:', error);
        return [];
    }
}

// Deprecated but kept for backward compatibility
function saveTasksToLocalStorage() {
    try {
        localStorage.setItem('tasks', JSON.stringify(tasks));
    } catch (error) {
        console.error('Error saving tasks to localStorage:', error);
    }
}

// Helper function to format date and time
function formatDateTime(isoString) {
    if (!isoString) return 'N/A';
    
    const date = new Date(isoString);
    return date.toLocaleString();
}

// Helper function to format duration
function formatDuration(ms) {
    if (ms < 0) return "0s";
    
    const seconds = Math.floor(ms / 1000);
    const minutes = Math.floor(seconds / 60);
    const hours = Math.floor(minutes / 60);
    
    if (hours > 0) {
        return `${hours}h ${minutes % 60}m ${seconds % 60}s`;
    } else if (minutes > 0) {
        return `${minutes}m ${seconds % 60}s`;
    } else {
        return `${seconds}s`;
    }
}

// Display tasks in the UI
function displayTasks(tasks) {
    const taskList = document.getElementById('taskList');
    
    if (tasks.length === 0) {
        taskList.innerHTML = '<p class="text-center text-muted py-4">No tasks found</p>';
        return;
    }
    
    // Sort tasks - most recent first
    tasks.sort((a, b) => new Date(b.created_at) - new Date(a.created_at));
    
    let html = '';
    
    for (const task of tasks) {
        const statusClass = `status-${task.status}`;
        let statusBadge = '';
        
        switch (task.status) {
            case 'pending':
                statusBadge = '<span class="badge bg-warning text-dark">Pending</span>';
                break;
            case 'processing':
                statusBadge = '<span class="badge bg-primary"><i class="bi bi-arrow-repeat spin me-1"></i>Processing</span>';
                break;
            case 'completed':
                statusBadge = '<span class="badge bg-success">Completed</span>';
                break;
            case 'failed':
                statusBadge = '<span class="badge bg-danger">Failed</span>';
                break;
            case 'not_found':
                statusBadge = '<span class="badge bg-secondary">Previous Session</span>';
                break;
        }
        
        let timeInfo = '';
        if (task.started_at && task.completed_at) {
            const started = new Date(task.started_at);
            const completed = new Date(task.completed_at);
            const durationMs = completed - started;
            const duration = formatDuration(durationMs);
            timeInfo = `<div class="text-muted small">Duration: ${duration}</div>`;
        } else if (task.started_at) {
            const started = new Date(task.started_at);
            const now = new Date();
            const durationMs = now - started;
            const duration = formatDuration(durationMs);
            timeInfo = `<div class="text-muted small">Running for: ${duration}</div>`;
        }
        
        const shortFolder = task.input_folder.split('\\').pop() || task.input_folder;
        
        // Add abort button for pending and processing tasks
        let actionButtons = `
            <button class="btn btn-sm btn-outline-primary view-details me-1"
                    data-task-id="${task.request_id}">
                View Details
            </button>
        `;
        
        // Show abort button for pending/processing tasks
        if (task.status === 'pending' || task.status === 'processing') {
            actionButtons += `
                <button class="btn btn-sm btn-danger abort-task me-1"
                        data-task-id="${task.request_id}">
                    Abort
                </button>
            `;
        }
        
        // Add remove button for all tasks except active ones
        if (task.status !== 'processing' && task.status !== 'pending') {
            actionButtons += `
                <button class="btn btn-sm btn-outline-danger remove-task"
                        data-task-id="${task.request_id}">
                    Remove
                </button>
            `;
        }
        
        html += `
            <div class="card task-card ${statusClass} mb-3">
                <div class="card-body py-3">
                    <div class="row align-items-center">
                        <div class="col-md-5">
                            <div class="d-flex flex-column">
                                <h6 class="mb-1 text-truncate" title="${task.input_folder}">
                                    ${shortFolder}
                                </h6>
                                <div class="d-flex align-items-center">
                                    <span class="text-muted small me-2">ID: ${task.request_id.substring(0, 8)}...</span>
                                    ${statusBadge}
                                </div>
                            </div>
                        </div>
                        <div class="col-md-3">
                            ${timeInfo}
                            <div class="text-muted small timestamp">
                                ${formatDateTime(task.created_at)}
                            </div>
                        </div>
                        <div class="col-md-4 text-end">
                            ${actionButtons}
                        </div>
                    </div>
                </div>
            </div>
        `;
    }
    
    taskList.innerHTML = html;
    
    // Add event listeners to view details buttons
    document.querySelectorAll('.view-details').forEach(button => {
        button.addEventListener('click', function() {
            const taskId = this.getAttribute('data-task-id');
            showTaskDetails(taskId);
        });
    });
    
    // Add event listeners to remove task buttons
    document.querySelectorAll('.remove-task').forEach(button => {
        button.addEventListener('click', function() {
            const taskId = this.getAttribute('data-task-id');
            removeTask(taskId);
        });
    });
    
    // Add event listeners to abort task buttons
    document.querySelectorAll('.abort-task').forEach(button => {
        button.addEventListener('click', function() {
            const taskId = this.getAttribute('data-task-id');
            abortTask(taskId);
        });
    });
    
    // No longer update results grid automatically when tasks update
    // This gives manual control via the refresh button
}


// Update status counts
function updateStatusCounts(tasks) {
    const counts = {
        pending: 0,
        processing: 0,
        completed: 0,
        failed: 0,
        not_found: 0
    };
    
    for (const task of tasks) {
        counts[task.status] = (counts[task.status] || 0) + 1;
    }
    
    document.getElementById('pendingCount').textContent = counts.pending;
    document.getElementById('processingCount').textContent = counts.processing;
    document.getElementById('completedCount').textContent = counts.completed;
    document.getElementById('failedCount').textContent = counts.failed;
}

// Initialize the app when DOM is loaded
document.addEventListener('DOMContentLoaded', function() {
    // Initialize modal
    taskModal = new bootstrap.Modal(document.getElementById('taskDetailModal'));
    
    // Load existing tasks
    tasks = loadTasksFromLocalStorage();
    
    // Initial update of UI based on loaded tasks
    displayTasks(tasks);
    updateStatusCounts(tasks);
    
    // Initial data refresh
    refreshTasks().then(() => {
        updateQueueStatus();
    });
    
    // Fetch device info
    fetchDeviceInfo();
    
    // Handle form submission
    document.getElementById('jobForm').addEventListener('submit', submitJob);
    
    // Handle refresh button
    document.getElementById('refreshTasks').addEventListener('click', function() {
        refreshTasks().then(() => {
            updateQueueStatus();
        });
    });
    
    // Handle clear tasks button
    document.getElementById('clearTasks').addEventListener('click', clearTasks);
    
    // Handle log buttons
    document.getElementById('refreshLogs').addEventListener('click', () => fetchLogs(currentLogLevel));
    document.getElementById('clearLogs').addEventListener('click', clearLogs);
    
    // Log level filter buttons
    document.getElementById('logAllLevel').addEventListener('click', (e) => {
        setActiveLogFilter(e.target);
        currentLogLevel = null;
        fetchLogs();
    });
    
    document.getElementById('logInfoLevel').addEventListener('click', (e) => {
        setActiveLogFilter(e.target);
        currentLogLevel = 'INFO';
        fetchLogs(currentLogLevel);
    });
    
    document.getElementById('logErrorLevel').addEventListener('click', (e) => {
        setActiveLogFilter(e.target);
        currentLogLevel = 'ERROR';
        fetchLogs(currentLogLevel);
    });
    
    document.getElementById('logWarningLevel').addEventListener('click', (e) => {
        setActiveLogFilter(e.target);
        currentLogLevel = 'WARNING';
        fetchLogs(currentLogLevel);
    });
    
    // Initial load
    refreshTasks();
    updateQueueStatus();
    fetchLogs();
    
    // Set up periodic refresh (but don't update results grid automatically)
    setInterval(() => {
        // Update tasks from local storage first
        refreshTasks().then(() => {
            updateQueueStatus();
            fetchLogs(currentLogLevel);
            fetchDeviceInfo();
            // No longer automatically refresh results grid
        });
    }, REFRESH_INTERVAL);
});

// Helper to set active state on log filter buttons
function setActiveLogFilter(activeButton) {
    // Remove active class from all filter buttons
    document.querySelectorAll('#logAllLevel, #logInfoLevel, #logErrorLevel, #logWarningLevel')
        .forEach(btn => {
            btn.classList.remove('active');
            
            // Reset all buttons to outline style
            if (btn.id === 'logAllLevel') {
                btn.className = 'btn btn-sm btn-outline-secondary log-filter-btn';
            } else if (btn.id === 'logInfoLevel') {
                btn.className = 'btn btn-sm btn-outline-info log-filter-btn';
            } else if (btn.id === 'logErrorLevel') {
                btn.className = 'btn btn-sm btn-outline-danger log-filter-btn';
            } else if (btn.id === 'logWarningLevel') {
                btn.className = 'btn btn-sm btn-outline-warning log-filter-btn';
            }
        });
    
    // Add active class to clicked button
    activeButton.classList.add('active');
    
    // Set solid button style based on the button type
    if (activeButton.id === 'logAllLevel') {
        activeButton.classList.remove('btn-outline-secondary');
        activeButton.classList.add('btn-secondary');
    } else if (activeButton.id === 'logInfoLevel') {
        activeButton.classList.remove('btn-outline-info');
        activeButton.classList.add('btn-info');
    } else if (activeButton.id === 'logErrorLevel') {
        activeButton.classList.remove('btn-outline-danger');
        activeButton.classList.add('btn-danger');
    } else if (activeButton.id === 'logWarningLevel') {
        activeButton.classList.remove('btn-outline-warning');
        activeButton.classList.add('btn-warning');
    }
}


// Submit a new job
async function submitJob(event) {
    event.preventDefault();
    
    const inputFolder = document.getElementById('inputFolder').value;
    const saveOverlay = document.getElementById('saveOverlay').checked;
    const useTta = document.getElementById('useTta').checked;
    
    const request = {
        input_folder: inputFolder,
        save_overlay: saveOverlay,
        use_tta: useTta
    };
    
    // Store the original button text outside the try block
    const submitButton = document.querySelector('#jobForm button[type="submit"]');
    const originalButtonText = submitButton.innerHTML;
    
    try {
        // Show loading indicator
        submitButton.innerHTML = '<span class="spinner-border spinner-border-sm me-2"></span>Processing...';
        submitButton.disabled = true;
        
        const response = await fetch(`${API_BASE_URL}/process`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(request)
        });
        
        // Check if response is ok before trying to parse JSON
        if (!response.ok) {
            const errorText = await response.text();
            throw new Error(errorText || `Server error: ${response.status}`);
        }
        
        const result = await response.json();
        
        // Add new task to local tasks array
        tasks.unshift(result);
        
        // Reset form
        document.getElementById('jobForm').reset();
        
        // Show success message and refresh UI
        alert('Job submitted successfully. Processing will begin shortly.');
        refreshTasks();
        updateQueueStatus();
        
    } catch (error) {
        console.error('Error submitting job:', error);
        alert(`Error submitting job: ${error.message}`);
    } finally {
        // Reset submit button using the variable from outside the try block
        submitButton.innerHTML = originalButtonText;
        submitButton.disabled = false;
    }
}

// Fetch and display logs from the server
async function fetchLogs(level = null) {
    try {
        let url = `${API_BASE_URL}/logs`;
        if (level) {
            url += `?level=${level}`;
        }
        
        const response = await fetch(url);
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        const data = await response.json();
        displayLogs(data.logs);
    } catch (error) {
        console.error('Error fetching logs:', error);
        const logContainer = document.getElementById('logContainer');
        logContainer.innerHTML = `<div class="alert alert-danger">Error loading logs: ${error.message}</div>`;
    }
}

// Display logs in the UI
function displayLogs(logs) {
    const logContainer = document.getElementById('logContainer');
    
    if (!logs || logs.length === 0) {
        logContainer.innerHTML = '<p class="text-center text-muted py-4">No logs available</p>';
        return;
    }
    
    let logHtml = '';
    logs.forEach(log => {
        const timestamp = new Date(log.timestamp).toLocaleTimeString();
        const date = new Date(log.timestamp).toLocaleDateString();
        
        // Add icon based on log level
        let levelIcon = '';
        switch(log.level) {
            case 'INFO':
                levelIcon = '<i class="bi bi-info-circle me-1"></i>';
                break;
            case 'ERROR':
                levelIcon = '<i class="bi bi-exclamation-triangle-fill me-1"></i>';
                break;
            case 'WARNING':
                levelIcon = '<i class="bi bi-exclamation-circle me-1"></i>';
                break;
            case 'DEBUG':
                levelIcon = '<i class="bi bi-bug me-1"></i>';
                break;
            default:
                levelIcon = '<i class="bi bi-chat-text me-1"></i>';
        }
        
        logHtml += `<div class="log-entry log-level-${log.level}">
            <span class="timestamp opacity-75">[${date} ${timestamp}]</span>
            <span class="log-level">${levelIcon}${log.level}</span>: 
            <span class="log-message">${log.message}</span>
        </div>`;
    });
    
    logContainer.innerHTML = logHtml;
    
    // Auto-scroll to bottom to show most recent logs
    logContainer.scrollTop = logContainer.scrollHeight;
}

// Clear logs display and from server
async function clearLogs() {
    try {
        const response = await fetch(`${API_BASE_URL}/logs/clear`, {
            method: 'POST'
        });
        
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        const logContainer = document.getElementById('logContainer');
        logContainer.innerHTML = '<p class="text-center text-muted">Logs cleared.</p>';
        
    } catch (error) {
        console.error('Error clearing logs:', error);
        alert(`Failed to clear logs: ${error.message}`);
    }
}

// Fetch and display all tasks
async function refreshTasks() {
    try {
        // Fetch all tasks from the API
        const response = await fetch(`${API_BASE_URL}/tasks?limit=50`);
        
        if (!response.ok) {
            throw new Error(`Failed to fetch tasks: ${response.statusText}`);
        }
        
        const taskData = await response.json();
        
        // Replace local tasks array with server-provided tasks
        tasks = taskData;
        
        // Display tasks
        displayTasks(tasks);
        
        // Update status counts
        updateStatusCounts(tasks);
        
        // Update UI
        document.getElementById('lastUpdated').textContent = formatDateTime(new Date().toISOString());
        
    } catch (error) {
        console.error('Error refreshing tasks:', error);
        
        // If API fails, try to fall back to localStorage as a last resort
        if (tasks.length === 0) {
            tasks = loadTasksFromLocalStorage();
            if (tasks.length > 0) {
                console.log('Loaded tasks from localStorage fallback');
                displayTasks(tasks);
                updateStatusCounts(tasks);
            }
        }
    }
}

// Update queue status
async function updateQueueStatus() {
    try {
        const response = await fetch(`${API_BASE_URL}/queue/length`);
        if (response.ok) {
            const data = await response.json();
            document.getElementById('queueCount').textContent = data.queue_size;
            document.getElementById('lastUpdated').textContent = new Date().toLocaleTimeString();
        }
    } catch (error) {
        console.error('Error updating queue status:', error);
        document.getElementById('queueCount').textContent = 'Error';
    }
}

// Function to clear task history
async function clearTasks() {
    if (!confirm('Are you sure you want to clear all task history? This cannot be undone.')) {
        return;
    }
    
    try {
        const response = await fetch(`${API_BASE_URL}/tasks`, {
            method: 'DELETE'
        });
        
        if (!response.ok) {
            throw new Error(`Failed to clear tasks: ${response.statusText}`);
        }
        
        // Clear local tasks array
        tasks = [];
        
        // Update UI
        displayTasks(tasks);
        updateStatusCounts(tasks);
        alert('Task history cleared successfully.');
        
    } catch (error) {
        console.error('Error clearing tasks:', error);
        alert(`Error clearing tasks: ${error.message}`);
    }
}

// Function to remove a task from the display
async function removeTask(taskId) {
    // Find the task to check its status
    const task = tasks.find(t => t.request_id === taskId);
    
    // Prevent removing active tasks
    if (task && (task.status === 'processing' || task.status === 'pending')) {
        alert('Cannot remove an active task. Please wait for it to complete or abort it first.');
        return;
    }
    
    if (!confirm('Are you sure you want to remove this task from the list?')) {
        return;
    }
    
    try {
        // Call API to remove task if it's available
        try {
            const response = await fetch(`${API_BASE_URL}/tasks/${taskId}`, {
                method: 'DELETE'
            });
            
            if (!response.ok) {
                // Check if it's because the task is active
                if (response.status === 400) {
                    const data = await response.json();
                    alert(data.detail || 'Cannot remove an active task');
                    return;
                }
                console.warn(`Could not remove task from server: ${response.statusText}`);
            }
        } catch (apiError) {
            console.warn('Could not remove task from server, removing from local display only:', apiError);
        }
        
        // Remove task from local array
        const taskIndex = tasks.findIndex(t => t.request_id === taskId);
        if (taskIndex !== -1) {
            tasks.splice(taskIndex, 1);
            
            // Update UI
            displayTasks(tasks);
            updateStatusCounts(tasks);
        }
        
    } catch (error) {
        console.error('Error removing task:', error);
        alert(`Error removing task: ${error.message}`);
    }
}

// Function to abort a running task
async function abortTask(taskId) {
    if (!confirm('Are you sure you want to abort this task? This will stop processing.')) {
        return;
    }
    
    try {
        const response = await fetch(`${API_BASE_URL}/tasks/${taskId}/abort`, {
            method: 'POST'
        });
        
        if (!response.ok) {
            throw new Error(`Failed to abort task: ${response.statusText}`);
        }
        
        alert('Task abort request sent. The task will be stopped soon.');
        
        // Refresh to show updated status
        setTimeout(() => {
            refreshTasks();
        }, 1000);
        
    } catch (error) {
        console.error('Error aborting task:', error);
        alert(`Error aborting task: ${error.message}`);
    }
}

// Show task details in modal
function showTaskDetails(taskId) {
    const task = tasks.find(t => t.request_id === taskId);
    
    if (!task) {
        console.error(`Task ${taskId} not found`);
        return;
    }
    
    let statusBadge = '';
    switch (task.status) {
        case 'pending':
            statusBadge = '<span class="badge bg-warning text-dark">Pending</span>';
            break;
        case 'processing':
            statusBadge = '<span class="badge bg-primary"><i class="bi bi-arrow-repeat spin me-1"></i>Processing</span>';
            break;
        case 'completed':
            statusBadge = '<span class="badge bg-success">Completed</span>';
            break;
        case 'failed':
            statusBadge = '<span class="badge bg-danger">Failed</span>';
            break;
        case 'not_found':
            statusBadge = '<span class="badge bg-secondary">Previous Session</span>';
            break;
    }
    
    let outputFolders = '';
    if (task.output_folders && task.output_folders.length > 0) {
        outputFolders = task.output_folders.map(folder => 
            `<div class="mb-2 text-break">
                <i class="bi bi-folder me-2"></i>${folder}
            </div>`
        ).join('');
    } else {
        outputFolders = '<div class="text-muted"><i class="bi bi-info-circle me-2"></i>No output folders yet</div>';
    }
    
    let errorMessage = '';
    if (task.error_message) {
        errorMessage = `
            <div class="alert alert-danger mt-3">
                <h6 class="mb-2"><i class="bi bi-exclamation-triangle me-2"></i>Error details:</h6>
                <pre class="mb-0 bg-light p-2 rounded" style="font-size: 0.85rem;">${task.error_message}</pre>
            </div>
        `;
    }
    
    // Calculate duration if available
    let durationStr = 'N/A';
    if (task.started_at && task.completed_at) {
        const started = new Date(task.started_at);
        const completed = new Date(task.completed_at);
        const durationMs = completed - started;
        durationStr = formatDuration(durationMs);
    } else if (task.started_at) {
        const started = new Date(task.started_at);
        const now = new Date();
        const durationMs = now - started;
        durationStr = `${formatDuration(durationMs)} (running)`;
    }
    
    let notFoundWarning = '';
    if (task.status === 'not_found') {
        notFoundWarning = `
            <div class="alert alert-warning mb-3">
                <h6 class="mb-0"><i class="bi bi-exclamation-circle me-2"></i>Task from previous session</h6>
                <p class="mb-0 mt-2">This task was created in a previous session and is no longer available on the server. 
                The server was likely restarted since this task was created.</p>
            </div>
        `;
    }
    
    // Determine which image to show based on save_overlay setting
    let imageTypeToShow = 'mask';
    if (task.save_overlay === true || task.save_overlay === undefined) {
        imageTypeToShow = 'overlay';
    }
    
    // Image preview section (only for completed tasks)
    let imagePreview = '';
    if (task.status === 'completed' || task.status === 'not_found') {
        // Determine which folder to look for images in
        const overlayFolder = task.output_folders ? task.output_folders.find(f => f.includes('overlay')) : null;
        const maskFolder = task.output_folders ? task.output_folders.find(f => f.includes('mask')) : null;
        
        // Create tabs for switching between overlay and mask
        imagePreview = `
            <div class="row mb-4">
                <div class="col-12">
                    <h6 class="mb-3"><i class="bi bi-card-image me-2"></i>Result Preview</h6>
                    <ul class="nav nav-tabs" id="resultTabs" role="tablist">
                        ${overlayFolder ? `
                            <li class="nav-item" role="presentation">
                                <button class="nav-link ${imageTypeToShow === 'overlay' ? 'active' : ''}" 
                                    id="overlay-tab" data-bs-toggle="tab" data-bs-target="#overlay-content" 
                                    type="button" role="tab">Overlay</button>
                            </li>
                        ` : ''}
                        ${maskFolder ? `
                            <li class="nav-item" role="presentation">
                                <button class="nav-link ${imageTypeToShow === 'mask' ? 'active' : ''}" 
                                    id="mask-tab" data-bs-toggle="tab" data-bs-target="#mask-content" 
                                    type="button" role="tab">Segmentation Mask</button>
                            </li>
                        ` : ''}
                    </ul>
                    <div class="tab-content p-3 border border-top-0 rounded-bottom" id="resultTabsContent">
                        ${overlayFolder ? `
                            <div class="tab-pane fade ${imageTypeToShow === 'overlay' ? 'show active' : ''}" 
                                id="overlay-content" role="tabpanel">
                                <div class="image-gallery">
                                    <div class="text-center image-container" id="overlay-image-container">
                                        <img src="${API_BASE_URL}/images/${task.request_id}/overlay/0" 
                                             class="img-fluid rounded" 
                                             alt="Overlay image" 
                                             onerror="this.onerror=null; this.src=''; this.alt='Image not available'; this.parentElement.classList.add('image-error');">
                                    </div>
                                    <div class="image-navigation d-flex justify-content-between align-items-center mt-3">
                                        <button class="btn btn-sm btn-outline-secondary prev-image" data-type="overlay" data-task="${task.request_id}" disabled>
                                            <i class="bi bi-chevron-left"></i> Previous
                                        </button>
                                        <div class="text-center">
                                            <small class="text-muted">From: ${overlayFolder}</small>
                                            <div class="image-counter mt-1" id="overlay-counter">Image 1 / ?</div>
                                        </div>
                                        <button class="btn btn-sm btn-outline-secondary next-image" data-type="overlay" data-task="${task.request_id}">
                                            Next <i class="bi bi-chevron-right"></i>
                                        </button>
                                    </div>
                                </div>
                            </div>
                        ` : ''}
                        ${maskFolder ? `
                            <div class="tab-pane fade ${imageTypeToShow === 'mask' ? 'show active' : ''}" 
                                id="mask-content" role="tabpanel">
                                <div class="image-gallery">
                                    <div class="text-center image-container" id="mask-image-container">
                                        <img src="${API_BASE_URL}/images/${task.request_id}/mask/0" 
                                             class="img-fluid rounded" 
                                             alt="Segmentation mask" 
                                             onerror="this.onerror=null; this.src=''; this.alt='Image not available'; this.parentElement.classList.add('image-error');">
                                    </div>
                                    <div class="image-navigation d-flex justify-content-between align-items-center mt-3">
                                        <button class="btn btn-sm btn-outline-secondary prev-image" data-type="mask" data-task="${task.request_id}" disabled>
                                            <i class="bi bi-chevron-left"></i> Previous
                                        </button>
                                        <div class="text-center">
                                            <small class="text-muted">From: ${maskFolder}</small>
                                            <div class="image-counter mt-1" id="mask-counter">Image 1 / ?</div>
                                        </div>
                                        <button class="btn btn-sm btn-outline-secondary next-image" data-type="mask" data-task="${task.request_id}">
                                            Next <i class="bi bi-chevron-right"></i>
                                        </button>
                                    </div>
                                </div>
                            </div>
                        ` : ''}
                    </div>
                </div>
            </div>
        `;
    }
    
    const detailContent = document.getElementById('taskDetailContent');
    detailContent.innerHTML = `
        ${notFoundWarning}
        <div class="mb-4">
            <h5 class="d-flex align-items-center mb-3">
                ${statusBadge}
                <span class="ms-2">Task ID: ${task.request_id}</span>
            </h5>
            <div class="text-muted">Created: ${formatDateTime(task.created_at)}</div>
        </div>
        
        ${imagePreview}
        
        <div class="row mb-4">
            <div class="col-md-6">
                <h6 class="mb-2"><i class="bi bi-folder-fill me-2"></i>Input Folder</h6>
                <div class="text-break bg-light p-2 rounded">${task.input_folder}</div>
            </div>
            <div class="col-md-6">
                <h6 class="mb-2"><i class="bi bi-stopwatch me-2"></i>Duration</h6>
                <div>${durationStr}</div>
            </div>
        </div>
        
        <div class="row mb-4">
            <div class="col-12">
                <h6 class="mb-2"><i class="bi bi-folder2-open me-2"></i>Output Folders</h6>
                <div class="p-2 bg-light rounded">
                    ${outputFolders}
                </div>
            </div>
        </div>
        
        <div class="row mb-3">
            <div class="col-md-4">
                <h6 class="mb-2"><i class="bi bi-calendar-event me-2"></i>Timeline</h6>
                <div class="small mb-2">
                    <i class="bi bi-circle-fill text-success me-2"></i>
                    Created: ${formatDateTime(task.created_at)}
                </div>
                <div class="small mb-2">
                    <i class="bi bi-circle-fill text-primary me-2"></i>
                    Started: ${task.started_at ? formatDateTime(task.started_at) : 'Not started'}
                </div>
                <div class="small">
                    <i class="bi bi-circle-fill text-danger me-2"></i>
                    Completed: ${task.completed_at ? formatDateTime(task.completed_at) : 'Not completed'}
                </div>
            </div>
        </div>
        
        ${errorMessage}
    `;
    
    taskModal.show();
    
    // Set up image navigation
    if (task.status === 'completed') {
        // Only initialize navigation for tasks that are not from previous sessions
        if (task.status !== 'not_found') {
            initializeImageNavigation('overlay', task.request_id);
            initializeImageNavigation('mask', task.request_id);
        } else {
            // For not_found tasks, update the image containers to show unavailable message
            document.querySelectorAll('.image-container').forEach(container => {
                container.classList.add('image-error');
                container.innerHTML = '<div class="text-center text-muted">Images not available from previous session</div>';
            });
        }
    }
}

// Initialize image navigation controls
function initializeImageNavigation(imageType, taskId) {
    // Image index tracking
    let currentIndex = 0;
    let totalImages = 0;
    let cameraFoldersList = [];
    let currentCameraIndex = -1;
    
    const container = document.getElementById(`${imageType}-image-container`);
    const counter = document.getElementById(`${imageType}-counter`);
    
    // If the container doesn't exist, return
    if (!container || !counter) return;
    
    // Set up event listeners for navigation buttons
    document.querySelectorAll(`.prev-image[data-type="${imageType}"]`).forEach(btn => {
        btn.addEventListener('click', function() {
            if (currentIndex > 0) {
                currentIndex--;
                updateImageDisplay();
            }
        });
    });
    
    document.querySelectorAll(`.next-image[data-type="${imageType}"]`).forEach(btn => {
        btn.addEventListener('click', function() {
            if (currentIndex < totalImages - 1) {
                currentIndex++;
                updateImageDisplay();
            }
        });
    });
    
    // Function to update the displayed image
    function updateImageDisplay() {
        const img = container.querySelector('img');
        if (img) {
            // Store the old src to detect if it's a new load
            const oldSrc = img.src;
            
            // Set the new image source
            img.src = `${API_BASE_URL}/images/${taskId}/${imageType}/${currentIndex}`;
            
            // If this is a new image load (src changed), fetch headers for camera info
            if (oldSrc !== img.src) {
                // Use fetch to get headers
                fetch(`${API_BASE_URL}/images/${taskId}/${imageType}/${currentIndex}`, {
                    method: 'HEAD'
                })
                .then(response => {
                    if (response.ok) {
                        // Get camera information from headers
                        const totalImagesHeader = response.headers.get('X-Total-Images');
                        const cameraFoldersHeader = response.headers.get('X-Camera-Folders');
                        const cameraCount = response.headers.get('X-Camera-Count');
                        const currentCamera = response.headers.get('X-Current-Camera');
                        const cameraIndex = response.headers.get('X-Camera-Index');
                        
                        if (totalImagesHeader) {
                            // Update total images count
                            totalImages = parseInt(totalImagesHeader);
                        }
                        
                        // Update camera folders list if available
                        if (cameraFoldersHeader) {
                            cameraFoldersList = cameraFoldersHeader.split(',');
                        }
                        
                        // Update current camera index if available
                        if (cameraIndex !== null) {
                            currentCameraIndex = parseInt(cameraIndex);
                        }
                        
                        // Update counter with camera information
                        if (currentCamera) {
                            if (cameraIndex !== null && cameraCount) {
                                // Show camera index out of total cameras
                                const camIdx = parseInt(cameraIndex) + 1;
                                const camCount = parseInt(cameraCount);
                                counter.innerHTML = `Camera ${camIdx} / ${camCount} <br><small class="text-muted">${currentCamera}</small>`;
                            } else {
                                // Just show the camera name
                                counter.innerHTML = `Image ${currentIndex + 1} / ${totalImages || '?'} <br><small class="text-muted">${currentCamera}</small>`;
                            }
                        } else {
                            // No camera info, just show image index
                            counter.textContent = `Image ${currentIndex + 1} / ${totalImages || '?'}`;
                        }
                        
                        // Update navigation controls
                        updateNavigationControls();
                    }
                })
                .catch(error => {
                    console.error('Error fetching image headers:', error);
                    // Fallback to basic counter
                    counter.textContent = `Image ${currentIndex + 1} / ${totalImages || '?'}`;
                    updateNavigationControls();
                });
            } else {
                // If src didn't change, just update the counter with existing info
                counter.textContent = `Image ${currentIndex + 1} / ${totalImages || '?'}`;
                updateNavigationControls();
            }
        }
    }
    
    // Function to update navigation controls based on current state
    function updateNavigationControls() {
        // Enable/disable previous button
        document.querySelectorAll(`.prev-image[data-type="${imageType}"]`).forEach(btn => {
            btn.disabled = currentIndex <= 0;
        });
        
        // Enable/disable next button
        document.querySelectorAll(`.next-image[data-type="${imageType}"]`).forEach(btn => {
            btn.disabled = currentIndex >= totalImages - 1 || totalImages === 0;
        });
    }
    
    // Try to get the first image to initialize
    const img = container.querySelector('img');
    if (img) {
        // When image loads, check for error and try to estimate total images
        img.addEventListener('load', function() {
            // Remove any error state
            container.classList.remove('image-error');
            
            // Image loaded successfully, start with 10 estimated images
            totalImages = 10;
            updateNavigationControls();
            
            // Try to find more images in background
            checkImageAvailability();
        });
        
        img.addEventListener('error', function() {
            // If first image fails, set total to 0
            totalImages = 0;
            updateNavigationControls();
            
            // Show error message in container
            container.classList.add('image-error');
            container.innerHTML = `
                <div class="text-center">
                    <div><i class="bi bi-exclamation-triangle-fill text-warning fs-3"></i></div>
                    <p class="text-muted mt-3">Could not load image</p>
                </div>
            `;
            
            // Try to get camera information from headers
            fetch(`${API_BASE_URL}/images/${taskId}/${imageType}/${currentIndex}`, {
                method: 'HEAD'
            })
            .then(response => {
                if (response.ok) {
                    const currentCamera = response.headers.get('X-Current-Camera');
                    const cameraIndex = response.headers.get('X-Camera-Index');
                    const cameraCount = response.headers.get('X-Camera-Count');
                    
                    if (currentCamera && cameraIndex !== null && cameraCount) {
                        const camIdx = parseInt(cameraIndex) + 1;
                        const camCount = parseInt(cameraCount);
                        counter.innerHTML = `Image not available <br><small class="text-muted">${currentCamera} (${camIdx}/${camCount})</small>`;
                    } else if (currentCamera) {
                        counter.innerHTML = `Image not available <br><small class="text-muted">${currentCamera}</small>`;
                    } else {
                        counter.textContent = 'Image not available';
                    }
                } else {
                    counter.textContent = 'Image not available';
                }
            })
            .catch(error => {
                console.error('Error fetching image headers:', error);
                counter.textContent = 'Image not available';
            });
            
            // Disable navigation buttons
            document.querySelectorAll(`.prev-image[data-type="${imageType}"], .next-image[data-type="${imageType}"]`).forEach(btn => {
                btn.disabled = true;
            });
        });
    }
    
    // Check how many images are available in background
    function checkImageAvailability() {
        // Make a HEAD request to the first image to get total count from headers
        fetch(`${API_BASE_URL}/images/${taskId}/${imageType}/0`, {
            method: 'HEAD'
        })
        .then(response => {
            if (response.ok) {
                const totalImagesHeader = response.headers.get('X-Total-Images');
                if (totalImagesHeader) {
                    totalImages = parseInt(totalImagesHeader);
                    updateNavigationControls();
                }
                
                // Get camera folders information
                const cameraFoldersHeader = response.headers.get('X-Camera-Folders');
                if (cameraFoldersHeader) {
                    cameraFoldersList = cameraFoldersHeader.split(',');
                }
            }
        })
        .catch(error => {
            console.error('Error checking image availability:', error);
        });
    }
}

// Function to update the runtime display for active tasks
function updateRuntimeDisplay() {
    // Only update if we have tasks
    if (tasks.length === 0) return;
    
    // Find all task cards with processing status
    const processingTaskCards = document.querySelectorAll('.status-processing');
    
    processingTaskCards.forEach(card => {
        // Find the task ID from the view details button
        const viewDetailsBtn = card.querySelector('.view-details');
        if (!viewDetailsBtn) return;
        
        const taskId = viewDetailsBtn.getAttribute('data-task-id');
        const task = tasks.find(t => t.request_id === taskId);
        
        if (task && task.started_at) {
            const started = new Date(task.started_at);
            const now = new Date();
            const durationMs = now - started;
            const duration = formatDuration(durationMs);
            
            // Find and update the runtime display
            const runtimeDiv = card.querySelector('.text-muted.small:not(.timestamp)');
            if (runtimeDiv) {
                runtimeDiv.innerHTML = `<div class="text-muted small">Running for: ${duration}</div>`;
            }
        }
    });
}

// Set up interval to update runtime display every 10 seconds
setInterval(updateRuntimeDisplay, 10000);


// Fetch and display device info
async function fetchDeviceInfo() {
    try {
        const response = await fetch(`${API_BASE_URL}/device/info`);
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        const data = await response.json();
        
        // Create device info HTML
        let deviceInfoHtml = `
            <div class="alert alert-info mb-0">
                <div class="d-flex align-items-center">
                    <i class="bi bi-gpu-card me-2 fs-4"></i>
                    <div>
                        <strong>Current Device:</strong> ${data.current_device}
                    </div>
                </div>`;
        
        // Add device properties if available
        if (data.device_properties && Object.keys(data.device_properties).length > 0) {
            const props = data.device_properties;
            deviceInfoHtml += `
                <div class="mt-2 small">
                    <div><strong>Name:</strong> ${props.name || 'N/A'}</div>
                    <div><strong>Memory:</strong> ${props.total_memory || 'N/A'}</div>
                </div>`;
        }
        
        deviceInfoHtml += `</div>`;
        
        // Add to the page
        const deviceInfoContainer = document.getElementById('deviceInfoContainer');
        if (deviceInfoContainer) {
            deviceInfoContainer.innerHTML = deviceInfoHtml;
        }
    } catch (error) {
        console.error('Error fetching device info:', error);
    }
}