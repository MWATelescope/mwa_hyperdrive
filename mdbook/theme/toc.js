// Add click handlers for sidebar menu toggle buttons
(function() {
    document.addEventListener('DOMContentLoaded', function() {
        // Find all toggle buttons in the sidebar
        var toggleButtons = document.querySelectorAll('.chapter li > a.toggle');

        toggleButtons.forEach(function(button) {
            button.addEventListener('click', function(e) {
                e.preventDefault();

                // Get the parent li element
                var li = button.parentElement;

                // Toggle the expanded class
                if (li.classList.contains('expanded')) {
                    li.classList.remove('expanded');
                } else {
                    li.classList.add('expanded');
                }
            });
        });

        // Also handle click on the toggle div specifically
        var toggleDivs = document.querySelectorAll('.chapter li > a.toggle div');
        toggleDivs.forEach(function(div) {
            div.style.cursor = 'pointer';
            div.addEventListener('click', function(e) {
                e.preventDefault();
                e.stopPropagation();
                var button = div.closest('a.toggle');
                if (button) {
                    button.click();
                }
            });
        });
    });
})();
