document.addEventListener('DOMContentLoaded', function() {
    const form = document.getElementById('burnoutForm');
    const resultDiv = document.getElementById('result');
    const predictionP = document.getElementById('prediction');
    const recommendationsUl = document.getElementById('recommendations');

    form.addEventListener('submit', function(event) {
        event.preventDefault();

        const formData = new FormData(form);
        const data = {
            sleep_hours: parseInt(formData.get('sleep_hours')),
            assignments: parseInt(formData.get('assignments')),
            mood: parseInt(formData.get('mood')),
            step_count: parseInt(formData.get('step_count')),
            heart_rate: parseInt(formData.get('heart_rate')),
            study_hours: parseInt(formData.get('study_hours'))
        };

        fetch('http://127.0.0.1:5000/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(data)
        })
        .then(response => response.json())
        .then(result => {
            // Update prediction text and class based on the result
            predictionP.textContent = `Prediction: ${result.prediction}`;
            predictionP.className = 'prediction'; // Reset class
            if (result.prediction.toLowerCase() === 'high') {
                predictionP.classList.add('high');
            } else if (result.prediction.toLowerCase() === 'medium') {
                predictionP.classList.add('medium');
            }

            recommendationsUl.innerHTML = '';
            result.recommendations.forEach(recommendation => {
                const li = document.createElement('li');
                li.textContent = recommendation;
                recommendationsUl.appendChild(li);
            });

            resultDiv.classList.add('show'); // Add the 'show' class for fade-in
        })
        .catch(error => {
            console.error('Error:', error);
            predictionP.textContent = 'Error predicting burnout risk.';
            predictionP.className = 'prediction high'; // Indicate error with high color
            recommendationsUl.innerHTML = '';
            resultDiv.classList.add('show');
        });
    });
});