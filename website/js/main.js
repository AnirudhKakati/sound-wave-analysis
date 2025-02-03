// Initialize AOS
AOS.init();

let isPlaying = false;
const music = document.getElementById('bgMusic');

// Check localStorage for music state on page load
document.addEventListener('DOMContentLoaded', function() {
    isPlaying = localStorage.getItem('musicPlaying') === 'true';
    const button = document.getElementById('musicToggle');
    
    if (!isPlaying) {
        button.classList.add('off');
        music.pause();
    }
});

function startExperience() {
    document.getElementById('splashScreen').style.opacity = '0';
    document.getElementById('mainContent').style.opacity = '1';
    setTimeout(() => {
        document.getElementById('splashScreen').style.display = 'none';
        music.play();
        isPlaying = true;
        localStorage.setItem('musicPlaying', true);
    }, 500);
}

function toggleMusic() {
    const button = document.getElementById('musicToggle');
    
    if (isPlaying) {
        music.pause();
        button.classList.add('off');
    } else {
        music.play();
        button.classList.remove('off');
    }
    isPlaying = !isPlaying;
    localStorage.setItem('musicPlaying', isPlaying);
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