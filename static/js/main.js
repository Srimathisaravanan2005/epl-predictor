// EPL Points Prediction - Main JavaScript

// Global variables
let currentChart = null;

// Initialize application
document.addEventListener('DOMContentLoaded', function() {
    initializeApp();
});

function initializeApp() {
    // Add fade-in animation to cards
    animateCards();
    
    // Initialize tooltips
    initializeTooltips();
    
    // Initialize smooth scrolling
    initializeSmoothScrolling();
    
    // Add loading states to buttons
    initializeLoadingStates();
    
    // Initialize form validations
    initializeFormValidations();
    
    console.log('EPL Predictor App Initialized');
}

// Animation functions
function animateCards() {
    const cards = document.querySelectorAll('.card');
    
    const observer = new IntersectionObserver((entries) => {
        entries.forEach((entry, index) => {
            if (entry.isIntersecting) {
                setTimeout(() => {
                    entry.target.classList.add('fade-in');
                }, index * 100);
            }
        });
    }, {
        threshold: 0.1
    });
    
    cards.forEach(card => {
        observer.observe(card);
    });
}

// Tooltip initialization
function initializeTooltips() {
    const tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
    tooltipTriggerList.map(function (tooltipTriggerEl) {
        return new bootstrap.Tooltip(tooltipTriggerEl);
    });
}

// Smooth scrolling
function initializeSmoothScrolling() {
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function (e) {
            e.preventDefault();
            const target = document.querySelector(this.getAttribute('href'));
            if (target) {
                target.scrollIntoView({
                    behavior: 'smooth',
                    block: 'start'
                });
            }
        });
    });
}

// Loading states for buttons
function initializeLoadingStates() {
    document.querySelectorAll('form').forEach(form => {
        form.addEventListener('submit', function() {
            const submitBtn = this.querySelector('button[type="submit"]');
            if (submitBtn) {
                const originalText = submitBtn.innerHTML;
                submitBtn.innerHTML = '<i class="fas fa-spinner fa-spin me-2"></i>Processing...';
                submitBtn.disabled = true;
                
                // Re-enable after 3 seconds (fallback)
                setTimeout(() => {
                    submitBtn.innerHTML = originalText;
                    submitBtn.disabled = false;
                }, 3000);
            }
        });
    });
}

// Form validations
function initializeFormValidations() {
    const forms = document.querySelectorAll('.needs-validation');
    
    Array.from(forms).forEach(form => {
        form.addEventListener('submit', event => {
            if (!form.checkValidity()) {
                event.preventDefault();
                event.stopPropagation();
            }
            form.classList.add('was-validated');
        });
    });
}

// Utility functions
function showNotification(message, type = 'info') {
    const notification = document.createElement('div');
    notification.className = `alert alert-${type} alert-dismissible fade show position-fixed`;
    notification.style.cssText = 'top: 20px; right: 20px; z-index: 9999; min-width: 300px;';
    notification.innerHTML = `
        ${message}
        <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
    `;
    
    document.body.appendChild(notification);
    
    // Auto-remove after 5 seconds
    setTimeout(() => {
        if (notification.parentNode) {
            notification.remove();
        }
    }, 5000);
}

function formatNumber(num, decimals = 0) {
    return Number(num).toLocaleString('en-US', {
        minimumFractionDigits: decimals,
        maximumFractionDigits: decimals
    });
}

function formatPercentage(num, decimals = 1) {
    return `${Number(num).toFixed(decimals)}%`;
}

// Chart utilities
function createPieChart(canvasId, data, labels, colors) {
    const ctx = document.getElementById(canvasId);
    if (!ctx) return null;
    
    // Destroy existing chart
    if (currentChart) {
        currentChart.destroy();
    }
    
    currentChart = new Chart(ctx, {
        type: 'pie',
        data: {
            labels: labels,
            datasets: [{
                data: data,
                backgroundColor: colors || ['#9c88ff', '#d1c4e9', '#f3e8ff'],
                borderWidth: 2,
                borderColor: '#fff'
            }]
        },
        options: {
            responsive: true,
            plugins: {
                legend: {
                    position: 'bottom',
                    labels: {
                        padding: 20,
                        usePointStyle: true
                    }
                },
                tooltip: {
                    callbacks: {
                        label: function(context) {
                            const percentage = ((context.parsed / context.dataset.data.reduce((a, b) => a + b, 0)) * 100).toFixed(1);
                            return `${context.label}: ${percentage}%`;
                        }
                    }
                }
            },
            animation: {
                animateRotate: true,
                duration: 1000
            }
        }
    });
    
    return currentChart;
}

function createBarChart(canvasId, data, labels, label = 'Data') {
    const ctx = document.getElementById(canvasId);
    if (!ctx) return null;
    
    // Destroy existing chart
    if (currentChart) {
        currentChart.destroy();
    }
    
    currentChart = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: labels,
            datasets: [{
                label: label,
                data: data,
                backgroundColor: '#9c88ff',
                borderColor: '#7c4dff',
                borderWidth: 1,
                borderRadius: 5
            }]
        },
        options: {
            responsive: true,
            scales: {
                y: {
                    beginAtZero: true,
                    grid: {
                        color: 'rgba(0,0,0,0.1)'
                    }
                },
                x: {
                    grid: {
                        display: false
                    }
                }
            },
            plugins: {
                legend: {
                    display: false
                },
                tooltip: {
                    backgroundColor: 'rgba(0,0,0,0.8)',
                    titleColor: '#fff',
                    bodyColor: '#fff',
                    borderColor: '#9c88ff',
                    borderWidth: 1
                }
            },
            animation: {
                duration: 1000,
                easing: 'easeOutQuart'
            }
        }
    });
    
    return currentChart;
}

// Prediction utilities
function validatePredictionInput(data) {
    const errors = [];
    const requiredFields = ['HS', 'AS', 'HST', 'AST', 'HC', 'AC', 'HF', 'AF', 'HY', 'AY', 'HR', 'AR'];
    
    requiredFields.forEach(field => {
        if (!(field in data)) {
            errors.push(`Missing field: ${field}`);
        } else {
            const value = parseFloat(data[field]);
            if (isNaN(value) || value < 0) {
                errors.push(`${field} must be a non-negative number`);
            }
        }
    });
    
    return errors;
}

function displayPredictionResult(prediction, probabilities) {
    const resultLabels = {
        'H': 'Home Win',
        'D': 'Draw',
        'A': 'Away Win'
    };
    
    const resultColors = {
        'H': '#28a745',
        'D': '#ffc107',
        'A': '#dc3545'
    };
    
    const resultContainer = document.getElementById('predictionResults');
    if (!resultContainer) return;
    
    const confidence = (probabilities[prediction] * 100).toFixed(1);
    
    resultContainer.innerHTML = `
        <div class="text-center mb-4">
            <div class="prediction-badge bg-${prediction === 'H' ? 'success' : prediction === 'D' ? 'warning' : 'danger'} text-white rounded-pill p-3 mb-3">
                <h4 class="mb-0">
                    <i class="fas fa-trophy me-2"></i>${resultLabels[prediction]}
                </h4>
                <p class="mb-0">Confidence: ${confidence}%</p>
            </div>
        </div>
        
        <div class="probability-breakdown">
            <h6 class="fw-bold mb-3">Probability Breakdown:</h6>
            ${Object.entries(probabilities).map(([key, value]) => `
                <div class="d-flex justify-content-between align-items-center mb-2">
                    <span>${resultLabels[key]}:</span>
                    <div class="d-flex align-items-center">
                        <div class="progress me-2" style="width: 100px; height: 8px;">
                            <div class="progress-bar" style="width: ${(value * 100).toFixed(1)}%; background-color: ${resultColors[key]};"></div>
                        </div>
                        <span class="fw-bold">${(value * 100).toFixed(1)}%</span>
                    </div>
                </div>
            `).join('')}
        </div>
    `;
    
    // Create pie chart
    const chartData = Object.values(probabilities);
    const chartLabels = Object.keys(probabilities).map(key => resultLabels[key]);
    const chartColors = Object.keys(probabilities).map(key => resultColors[key]);
    
    setTimeout(() => {
        createPieChart('probabilityChart', chartData, chartLabels, chartColors);
    }, 100);
}

// Team comparison utilities
function compareTeams(team1Stats, team2Stats) {
    const comparison = {};
    
    Object.keys(team1Stats).forEach(stat => {
        const value1 = parseFloat(team1Stats[stat]) || 0;
        const value2 = parseFloat(team2Stats[stat]) || 0;
        
        comparison[stat] = {
            team1: value1,
            team2: value2,
            winner: value1 > value2 ? 'team1' : value2 > value1 ? 'team2' : 'tie'
        };
    });
    
    return comparison;
}

// Data export utilities
function exportToCSV(data, filename = 'epl_data.csv') {
    const csv = convertToCSV(data);
    const blob = new Blob([csv], { type: 'text/csv' });
    const url = window.URL.createObjectURL(blob);
    
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    a.click();
    
    window.URL.revokeObjectURL(url);
}

function convertToCSV(data) {
    if (!data || data.length === 0) return '';
    
    const headers = Object.keys(data[0]);
    const csvContent = [
        headers.join(','),
        ...data.map(row => headers.map(header => {
            const value = row[header];
            return typeof value === 'string' && value.includes(',') ? `"${value}"` : value;
        }).join(','))
    ].join('\n');
    
    return csvContent;
}

// Local storage utilities
function saveToLocalStorage(key, data) {
    try {
        localStorage.setItem(key, JSON.stringify(data));
        return true;
    } catch (error) {
        console.error('Error saving to localStorage:', error);
        return false;
    }
}

function loadFromLocalStorage(key) {
    try {
        const data = localStorage.getItem(key);
        return data ? JSON.parse(data) : null;
    } catch (error) {
        console.error('Error loading from localStorage:', error);
        return null;
    }
}

// API utilities
async function makeAPIRequest(url, options = {}) {
    try {
        const response = await fetch(url, {
            headers: {
                'Content-Type': 'application/json',
                ...options.headers
            },
            ...options
        });
        
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        return await response.json();
    } catch (error) {
        console.error('API request failed:', error);
        throw error;
    }
}

// Theme utilities
function toggleTheme() {
    const body = document.body;
    const isDark = body.classList.contains('dark-theme');
    
    if (isDark) {
        body.classList.remove('dark-theme');
        localStorage.setItem('theme', 'light');
    } else {
        body.classList.add('dark-theme');
        localStorage.setItem('theme', 'dark');
    }
}

function loadTheme() {
    const savedTheme = localStorage.getItem('theme');
    if (savedTheme === 'dark') {
        document.body.classList.add('dark-theme');
    }
}

// Performance monitoring
function measurePerformance(name, fn) {
    const start = performance.now();
    const result = fn();
    const end = performance.now();
    
    console.log(`${name} took ${(end - start).toFixed(2)} milliseconds`);
    return result;
}

// Error handling
window.addEventListener('error', function(event) {
    console.error('Global error:', event.error);
    showNotification('An unexpected error occurred. Please try again.', 'danger');
});

// Keyboard shortcuts
document.addEventListener('keydown', function(event) {
    // Ctrl/Cmd + K for search (if implemented)
    if ((event.ctrlKey || event.metaKey) && event.key === 'k') {
        event.preventDefault();
        const searchInput = document.querySelector('input[type="search"]');
        if (searchInput) {
            searchInput.focus();
        }
    }
    
    // Escape to close modals
    if (event.key === 'Escape') {
        const openModal = document.querySelector('.modal.show');
        if (openModal) {
            const modal = bootstrap.Modal.getInstance(openModal);
            if (modal) {
                modal.hide();
            }
        }
    }
});

// Export functions for global use
window.EPLPredictor = {
    showNotification,
    formatNumber,
    formatPercentage,
    createPieChart,
    createBarChart,
    validatePredictionInput,
    displayPredictionResult,
    compareTeams,
    exportToCSV,
    saveToLocalStorage,
    loadFromLocalStorage,
    makeAPIRequest,
    toggleTheme,
    measurePerformance
};

console.log('EPL Predictor JavaScript loaded successfully');