// ------------------------------------
// THEME TOGGLE
// ------------------------------------

function toggleTheme(){

    document
        .getElementById("body")
        .classList
        .toggle("light-mode");
}

// ------------------------------------
// LOADING BUTTON
// ------------------------------------

const form = document.querySelector("form");

form.addEventListener("submit", function(){

    const btn = document.getElementById(
        "analyzeBtn"
    );

    const btnText = document.getElementById(
        "btnText"
    );

    btn.disabled = true;

    btnText.innerHTML =
        "Analyzing...";
});

console.log(
    "Email Spam Classifier Loaded Successfully"
);