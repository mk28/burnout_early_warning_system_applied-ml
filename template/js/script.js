<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Login - Burnout Early Warning System</title>
    <link rel="stylesheet" href="style.css"> </head>
<body>
    <canvas id="background-canvas"></canvas>
    <div class="container">
        <div class="login-container">
            <h1>Welcome Back</h1>
            <form id="login-form" class="login-form" action="/login" method="POST">
                <div class="form-group">
                    <label for="username">Username:</label>
                    <input type="text" id="username" name="username" required>
                </div>
                <div class="form-group">
                    <label for="password">Password:</label>
                    <input type="password" id="password" name="password" required>
                </div>
                <button type="submit">Sign In</button>
            </form>
        </div>
    </div>
    <script>
        const canvas = document.getElementById('background-canvas');
        const ctx = canvas.getContext('2d');
        canvas.width = window.innerWidth;
        canvas.height = window.innerHeight;

        let particles = [];
        const particleCount = 120;
        const baseRadius = 5;
        const radiusVariation = 2;
        const baseSpeed = 0.2;
        const speedVariation = 0.1;
        const baseAlpha = 0.5;
        const alphaVariation = 0.2;
        const baseHue = 240;
        const hueVariation = 20;
        const trailLength = 20;
        const waveFrequency = 0.01;
        let time = 0;

        class Particle {
            constructor() {
                this.x = Math.random() * canvas.width;
                this.y = Math.random() * canvas.height;
                this.radius = baseRadius + Math.random() * radiusVariation;
                this.speedX = (Math.random() - 0.5) * (baseSpeed + Math.random() * speedVariation);
                this.speedY = (Math.random() - 0.5) * (baseSpeed + Math.random() * speedVariation);
                this.alpha = baseAlpha + Math.random() * alphaVariation;
                this.hue = baseHue + Math.random() * hueVariation;
                this.trail = [];
                this.initialY = this.y;
            }

            update() {
                this.x += this.speedX;
                this.y += this.speedY + Math.sin(time * waveFrequency + this.initialY * 0.02) * 0.5;

                if (this.x > canvas.width) this.x = 0;
                if (this.x < 0) this.x = canvas.width;
                if (this.y > canvas.height) this.y = 0;
                if (this.y < 0) this.y = canvas.height;

                this.trail.push({ x: this.x, y: this.y });
                if (this.trail.length > trailLength) {
                    this.trail.shift();
                }
            }

            draw() {
                for (let i = 0; i < this.trail.length; i++) {
                    const point = this.trail[i];
                    const opacity = this.alpha * (0.2 + (i / this.trailLength) * 0.8);
                    const size = this.radius * (0.7 + (i / this.trailLength) * 0.3);

                    ctx.beginPath();
                    ctx.arc(point.x, point.y, size, 0, Math.PI * 2);
                    ctx.fillStyle = `hsla(${this.hue}, 80%, 60%, ${opacity})`;
                    ctx.fill();
                    ctx.closePath();
                }
            }
        }

        function initParticles() {
            particles = [];
            for (let i = 0; i < particleCount; i++) {
                particles.push(new Particle());
            }
        }

        function animate() {
            ctx.clearRect(0, 0, canvas.width, canvas.height);
            time += 0.1;

            for (let i = 0; i < particles.length; i++) {
                particles[i].update();
                particles[i].draw();
            }

            requestAnimationFrame(animate);
        }

        initParticles();
        animate();

        window.addEventListener('resize', () => {
            canvas.width = window.innerWidth;
            canvas.height = window.innerHeight;
            initParticles();
        });

        // Get the form element
        const loginForm = document.getElementById('login-form');

        // Add an event listener to the form's submit event
        loginForm.addEventListener('submit', (event) => {
            event.preventDefault(); // Prevent the default form submission

            // Get the username and password values
            const username = loginForm.username.value;
            const password = loginForm.password.value;

            // In a real application, you would send this data to a server for validation
            // and handle the response.  For this example, we'll just simulate a successful login.
            if (username && password) {
                // Redirect to the index.html page
                window.location.href = 'index.html'; // Make sure this is the correct filename
            } else {
                alert('Invalid username or password.  Please try again.');
            }
        });
    </script>
</body>
</html>
