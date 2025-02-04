// Initialize AOS
AOS.init();

// Check page state on load
document.addEventListener('DOMContentLoaded', function() {
    const hasSeenPrompt = sessionStorage.getItem('hasSeenPrompt');
    const splashScreen = document.getElementById('splashScreen');
    const mainContent = document.getElementById('mainContent');
    
    if (hasSeenPrompt) {
        // Hide splash screen and show main content immediately
        splashScreen.style.display = 'none';
        mainContent.style.opacity = '1';
    } else {
        // First visit - show splash screen
        mainContent.style.opacity = '0';
    }
});

function startExperience() {
    // Mark that user has seen the prompt
    sessionStorage.setItem('hasSeenPrompt', 'true');
    
    // Hide splash screen and show main content
    const splashScreen = document.getElementById('splashScreen');
    const mainContent = document.getElementById('mainContent');
    
    splashScreen.style.opacity = '0';
    mainContent.style.opacity = '1';
    
    setTimeout(() => {
        splashScreen.style.display = 'none';
    }, 500);
}

// Smooth scrolling for navigation
document.querySelectorAll('a[href^="#"]').forEach(anchor => {
    anchor.addEventListener('click', function (e) {
        e.preventDefault();
        document.querySelector(this.getAttribute('href')).scrollIntoView({
            behavior: 'smooth'
        });
    });
});