// PCA Visualization handling

// Global variable to store PCA data
let pcaData = null;

// // Current view dimension (2d or 3d)
// let currentDimension = '3d';

// Chart instance
let pcaChart = null;

// Load PCA data
async function loadPCAData() {
    try {
        const response = await fetch('data/pca_data.json');
        if (!response.ok) {
            throw new Error(`Failed to load PCA data: ${response.status}`);
        }
        pcaData = await response.json();
        console.log('PCA data loaded successfully');
        return true;
    } catch (error) {
        console.error('Error loading PCA data:', error);
        document.getElementById('pcaVisualization').innerHTML = `
            <div style="padding: 2rem; text-align: center; color: #ff6b6b;">
                <p>Error loading PCA data. Please make sure you've generated the data file.</p>
                <p>Error details: ${error.message}</p>
            </div>
        `;
        return false;
    }
}

// Initialize category checkboxes
function initCategoryCheckboxes() {
    // Select/Deselect All buttons
    document.getElementById('selectAll').addEventListener('click', () => {
        document.querySelectorAll('input[name="category"]').forEach(checkbox => {
            checkbox.checked = true;
        });
    });

    document.getElementById('deselectAll').addEventListener('click', () => {
        document.querySelectorAll('input[name="category"]').forEach(checkbox => {
            checkbox.checked = false;
        });
    });

    // Pre-select a few categories by default
    const defaultCategories = ['applause', 'birds', 'speech'];
    defaultCategories.forEach(category => {
        const checkbox = document.querySelector(`input[value="${category}"]`);
        if (checkbox) checkbox.checked = true;
    });
}

// // Initialize dimension tabs
// function initDimensionTabs() {
//     document.querySelectorAll('.dimension-tab').forEach(tab => {
//         tab.addEventListener('click', () => {
//             // Remove active class from all tabs
//             document.querySelectorAll('.dimension-tab').forEach(t => {
//                 t.classList.remove('active');
//             });
            
//             // Add active class to clicked tab
//             tab.classList.add('active');
            
//             // Update current dimension
//             currentDimension = tab.dataset.dimension;
            
//             // Update visualization if we have selected categories
//             const selectedCategories = getSelectedCategories();
//             if (selectedCategories.length > 0) {
//                 generateVisualization(selectedCategories, parseInt(document.getElementById('samplesPerCategory').value));
//             }
//         });
//     });
// }

// Generate button event
function initGenerateButton() {
    document.getElementById('generatePCA').addEventListener('click', () => {
        const selectedCategories = getSelectedCategories();
        if (selectedCategories.length === 0) {
            alert('Please select at least one category.');
            return;
        }
        
        const samplesPerCategory = parseInt(document.getElementById('samplesPerCategory').value);
        if (isNaN(samplesPerCategory) || samplesPerCategory < 1 || samplesPerCategory > 750) {
            alert('Please enter a valid number of samples (1-750).');
            return;
        }
        
        generateVisualization(selectedCategories, samplesPerCategory);
    });
}

// Get selected categories
function getSelectedCategories() {
    const checkboxes = document.querySelectorAll('input[name="category"]:checked');
    return Array.from(checkboxes).map(checkbox => checkbox.value);
}

// Generate PCA visualization
// function generateVisualization(categories, samplesPerCategory) {
//     if (!pcaData) {
//         alert('PCA data is not loaded yet. Please wait or reload the page.');
//         return;
//     }
    
//     // Update selected categories display
//     updateSelectedCategoriesDisplay(categories);
    
//     // Generate the appropriate chart based on dimension
//     if (currentDimension === '2d') {
//         generate2DVisualization(categories, samplesPerCategory);
//     } else {
//         generate3DVisualization(categories, samplesPerCategory);
//     }
// }

function generateVisualization(categories, samplesPerCategory) {
    if (!pcaData) {
        alert('PCA data is not loaded yet. Please wait or reload the page.');
        return;
    }
    
    // Update selected categories display
    updateSelectedCategoriesDisplay(categories);
    
    // Generate 3D visualization
    generate3DVisualization(categories, samplesPerCategory);
}



// Update selected categories display
function updateSelectedCategoriesDisplay(categories) {
    const displayElement = document.getElementById('selectedCategoriesDisplay');
    
    if (categories.length === 0) {
        displayElement.innerHTML = '<p>No categories selected yet.</p>';
        return;
    }
    
    const formattedCategories = categories.map(cat => {
        return `<span class="category-tag">${cat.replace(/_/g, ' ')}</span>`;
    }).join(' ');
    
    displayElement.innerHTML = `
        <p>Selected categories: ${formattedCategories}</p>
        <p>Samples per category: ${document.getElementById('samplesPerCategory').value}</p>
    `;
}

// // Generate 2D visualization
// function generate2DVisualization(categories, samplesPerCategory) {
//     const visualizationElement = document.getElementById('pcaVisualization');
    
//     // Destroy existing chart if any
//     if (pcaChart) {
//         pcaChart.destroy();
//     }
    
//     // Prepare data for Chart.js
//     const datasets = [];
    
//     // Color palette for categories
//     const colorPalette = [
//         '#4285F4', '#EA4335', '#FBBC05', '#34A853', '#FF6D01', 
//         '#46BDC6', '#7E57C2', '#EC407A', '#AB47BC', '#EF5350',
//         '#66BB6A', '#FFA726', '#26C6DA', '#42A5F5', '#EC407A',
//         '#5C6BC0', '#78909C', '#8D6E63', '#9CCC65', '#29B6F6'
//     ];
    
//     // Create a dataset for each category
//     categories.forEach((category, index) => {
//         const categoryData = pcaData[category];
//         if (!categoryData) {
//             console.error(`No data found for category: ${category}`);
//             return;
//         }
        
//         // Sample the requested number of points
//         const allPoints = categoryData.PC1.length;
//         const indices = [];
        
//         // Generate random indices without replacement
//         if (samplesPerCategory < allPoints) {
//             // Random sampling
//             const indexPool = Array.from({length: allPoints}, (_, i) => i);
//             for (let i = 0; i < samplesPerCategory; i++) {
//                 const randomIndex = Math.floor(Math.random() * indexPool.length);
//                 indices.push(indexPool[randomIndex]);
//                 indexPool.splice(randomIndex, 1);
//             }
//         } else {
//             // If requesting more samples than available, use all
//             indices.push(...Array.from({length: allPoints}, (_, i) => i));
//         }
        
//         // Create data points using sampled indices
//         const dataPoints = indices.map(i => ({
//             x: categoryData.PC1[i],
//             y: categoryData.PC2[i]
//         }));
        
//         datasets.push({
//             label: category.replace(/_/g, ' '),
//             data: dataPoints,
//             backgroundColor: colorPalette[index % colorPalette.length],
//             borderColor: colorPalette[index % colorPalette.length],
//             pointRadius: 6,
//             pointHoverRadius: 8
//         });
//     });
    
//     // Create the chart
//     pcaChart = new Chart(visualizationElement, {
//         type: 'scatter',
//         data: {
//             datasets: datasets
//         },
//         options: {
//             responsive: true,
//             maintainAspectRatio: false,
//             plugins: {
//                 title: {
//                     display: true,
//                     text: '2D PCA Visualization',
//                     font: {
//                         size: 18,
//                         weight: 'bold'
//                     },
//                     padding: {
//                         top: 10,
//                         bottom: 30
//                     },
//                     color: '#000000'  // Change title color to black
//                 },
//                 legend: {
//                     labels: {
//                         color: '#000000' // Change legend text color to black
//                     }
//                 }
//             },
//             scales: {
//                 x: {
//                     title: {
//                         color: '#000000'  // Change X-axis label color to black
//                     },
//                     ticks: {
//                         color: '#000000'  // Change X-axis tick color to black
//                     },
//                     grid: {
//                         color: 'rgba(0, 0, 0, 0.1)'  // Change X-axis grid to light gray
//                     }
//                 },
//                 y: {
//                     title: {
//                         color: '#000000'  // Change Y-axis label color to black
//                     },
//                     ticks: {
//                         color: '#000000'  // Change Y-axis tick color to black
//                     },
//                     grid: {
//                         color: 'rgba(0, 0, 0, 0.1)'  // Change Y-axis grid to light gray
//                     }
//                 }
//             },
//             layout: {
//                 backgroundColor: '#ffffff' // Set full chart background to white
//             }
//         }
        
//     });
// }

// Generate 3D visualization
function generate3DVisualization(categories, samplesPerCategory) {
    const visualizationElement = document.getElementById('plotly3dContainer');
    
    // // For 3D visualization, we'll use Plotly.js
    // // Create a container div for Plotly
    // visualizationElement.innerHTML = '<div id="plotly3dContainer" style="width: 100%; height: 100%;"></div>';
    const plotlyContainer = document.getElementById('plotly3dContainer');
    
    
    // Clear the container if needed
    visualizationElement.innerHTML = '';
    
    // Color palette for categories
    const colorPalette = [
        '#4285F4', '#EA4335', '#FBBC05', '#34A853', '#FF6D01', 
        '#46BDC6', '#7E57C2', '#EC407A', '#AB47BC', '#EF5350',
        '#66BB6A', '#FFA726', '#26C6DA', '#42A5F5', '#EC407A',
        '#5C6BC0', '#78909C', '#8D6E63', '#9CCC65', '#29B6F6'
    ];
    
    // Prepare data for Plotly
    const traces = [];
    
    // Create a trace for each category
    categories.forEach((category, index) => {
        const categoryData = pcaData[category];
        if (!categoryData) {
            console.error(`No data found for category: ${category}`);
            return;
        }
        
        // Sample the requested number of points
        const allPoints = categoryData.PC1.length;
        const indices = [];

        // Generate random indices without replacement
        if (samplesPerCategory < allPoints) {
            // Random sampling
            const indexPool = Array.from({length: allPoints}, (_, i) => i);
            for (let i = 0; i < samplesPerCategory; i++) {
                const randomIndex = Math.floor(Math.random() * indexPool.length);
                indices.push(indexPool[randomIndex]);
                indexPool.splice(randomIndex, 1);
            }
        } else {
            // If requesting more samples than available, use all
            indices.push(...Array.from({length: allPoints}, (_, i) => i));
        }

        // Create arrays for the plot using sampled indices
        const x = indices.map(i => categoryData.PC1[i]);
        const y = indices.map(i => categoryData.PC2[i]);
        const z = indices.map(i => categoryData.PC3[i]);
        
        traces.push({
            type: 'scatter3d',
            mode: 'markers',
            name: category.replace(/_/g, ' '),
            x: x,
            y: y,
            z: z,
            marker: {
                size: 5,
                color: colorPalette[index % colorPalette.length],
                opacity: 0.8
            }
        });
    });
    
    // Create the 3D scatter plot
    Plotly.newPlot(plotlyContainer, traces, {
        title: {
            text:'<b>3D PCA Visualization</b>',
            font:{
                size: 30, 
                family: 'Arial, sans-serif',
                weight: 'bold'
            },
            pad: {
                t: 5,
                b: 2
            }
        },
        scene: {
            xaxis: { title: 'PC 1' },
            yaxis: { title: 'PC 2' },
            zaxis: { title: 'PC 3' }
        },
        margin: {
            l: 0,
            r: 0,
            b: 20,
            t: 80
        },
        legend: {
            font: {
                size: 16,  // Increase this value to make legend text larger
                family: 'Arial, sans-serif'
            },
            itemsizing: 'constant', // this is the key property for marker size in legend
            itemwidth: 20,
            y: 0.8,
            x: 0.8  
        },
        autosize: true,
        paper_bgcolor: 'rgb(255, 255, 255)',
        plot_bgcolor: 'rgb(0, 0, 0)',
        font: {
            color: '#000000'
        },
        hoverlabel: {
            bgcolor: '#000000',
            font: {
                color: '#ffffff'
            }
        }
    });
}

// Initialize when document is loaded
document.addEventListener('DOMContentLoaded', async function() {
    // Load Chart.js and Plotly.js from CDN
    await loadScripts();
    
    // Load PCA data
    const dataLoaded = await loadPCAData();
    if (!dataLoaded) return;
    
    // Initialize UI components
    initCategoryCheckboxes();
    // Remove initDimensionTabs() call
    initGenerateButton();
    
    // Generate visualization with default selections
    const defaultCategories = ['applause', 'birds', 'speech'];
    generateVisualization(defaultCategories, 15);
});

// Helper to load required scripts
async function loadScripts() {
    const scripts = [
        'https://cdnjs.cloudflare.com/ajax/libs/plotly.js/2.16.1/plotly.min.js'
    ];
    
    const loadPromises = scripts.map(src => {
        return new Promise((resolve, reject) => {
            const script = document.createElement('script');
            script.src = src;
            script.onload = resolve;
            script.onerror = reject;
            document.head.appendChild(script);
        });
    });
    
    try {
        await Promise.all(loadPromises);
        console.log('All required scripts loaded successfully');
    } catch (error) {
        console.error('Failed to load required scripts:', error);
        alert('Failed to load required visualization libraries. Please check your internet connection and reload the page.');
    }
}