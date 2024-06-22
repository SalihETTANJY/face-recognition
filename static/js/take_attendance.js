// take_attendance.js
document.addEventListener("DOMContentLoaded", function() {
    const video = document.getElementById('video');
    const canvas = document.getElementById('canvas');
    const captureButton = document.getElementById('capture');
    const resultat = document.getElementById('resultat');

    // Accéder à la webcam
    navigator.mediaDevices.getUserMedia({ video: true })
        .then(stream => {
            video.srcObject = stream;
        })
        .catch(err => {
            console.error("Erreur d'accès à la webcam : ", err);
        });

    captureButton.addEventListener('click', function() {
        canvas.getContext('2d').drawImage(video, 0, 0, canvas.width, canvas.height);
        const dataUrl = canvas.toDataURL('image/jpeg');

        // Envoyer l'image au serveur
        fetch('/process_attendance', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ image: dataUrl }),
        })
        .then(response => response.json())
        .then(data => {
            resultat.textContent = 'Présence prise pour: ' + data.name;
        })
        .catch(error => {
            console.error('Erreur:', error);
        });
    });
});
