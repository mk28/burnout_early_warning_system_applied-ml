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
            let predictionText = "";
            let predictionIcon = "";
            switch (result.prediction.toLowerCase()) {
                case 'low':
                    predictionText = "Low Risk - Keep up the good work!";
                    predictionIcon = "<i class='fas fa-check-circle'></i>"; // Example icon
                    predictionP.className = 'prediction low';
                    break;
                case 'medium':
                    predictionText = "Medium Risk - Time for some self-care.";
                    predictionIcon = "<i class='fas fa-exclamation-triangle'></i>"; // Example icon
                    predictionP.className = 'prediction medium';
                    break;
                case 'high':
                    predictionText = "High Risk - Let's focus on well-being.";
                    predictionIcon = "<i class='fas fa-heartbeat'></i>"; // Example icon
                    predictionP.className = 'prediction high';
                    break;
                default:
                    predictionText = `Prediction: ${result.prediction}`;
                    predictionP.className = 'prediction';
            }

            predictionP.innerHTML = `${predictionIcon} ${predictionText}`; // Combine icon and text

            recommendationsUl.innerHTML = '';
            result.recommendations.forEach(recommendation => {
                const li = document.createElement('li');
                li.innerHTML = `<i class="fas fa-arrow-right"></i> ${recommendation}`; // Example icon
                recommendationsUl.appendChild(li);
            });

            resultDiv.classList.add('show');
        })
        .catch(error => {
            console.error('Error:', error);
            predictionP.textContent = 'Error predicting burnout risk.';
            predictionP.className = 'prediction high';
            recommendationsUl.innerHTML = '';
            resultDiv.classList.add('show');
        });
    });
});