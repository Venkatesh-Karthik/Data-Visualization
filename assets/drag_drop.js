/**
 * Enhanced Drag & Drop Upload Interactions
 * Provides visual feedback for drag enter, over, leave, and drop events
 */

document.addEventListener('DOMContentLoaded', function() {
    // Wait for Dash to render the upload component
    setTimeout(initializeDragDrop, 100);
});

function initializeDragDrop() {
    const uploadDiv = document.getElementById('upload-data');
    
    if (!uploadDiv) {
        // Retry if not found
        setTimeout(initializeDragDrop, 100);
        return;
    }
    
    // Ensure pointer events are enabled
    uploadDiv.style.pointerEvents = 'auto';
    
    // Drag enter - initial glow
    uploadDiv.addEventListener('dragenter', function(e) {
        e.preventDefault();
        e.stopPropagation();
        this.classList.add('drag-enter');
        this.classList.remove('drag-leave');
    });
    
    // Drag over - stronger glow
    uploadDiv.addEventListener('dragover', function(e) {
        e.preventDefault();
        e.stopPropagation();
        this.classList.add('drag-over');
    });
    
    // Drag leave - remove effects
    uploadDiv.addEventListener('dragleave', function(e) {
        e.preventDefault();
        e.stopPropagation();
        
        // Only remove if we're leaving the upload div itself, not a child
        if (e.target === this || !this.contains(e.relatedTarget)) {
            this.classList.remove('drag-enter', 'drag-over');
            this.classList.add('drag-leave');
            
            // Remove drag-leave class after animation
            setTimeout(() => {
                this.classList.remove('drag-leave');
            }, 400);
        }
    });
    
    // Drop - success animation
    uploadDiv.addEventListener('drop', function(e) {
        e.preventDefault();
        e.stopPropagation();
        this.classList.remove('drag-enter', 'drag-over');
        this.classList.add('drop-success');
        
        // Remove success class after animation
        setTimeout(() => {
            this.classList.remove('drop-success');
        }, 1000);
    });
    
    // Prevent default drag behavior on document
    document.addEventListener('dragover', function(e) {
        e.preventDefault();
    });
    
    document.addEventListener('drop', function(e) {
        e.preventDefault();
    });
}
