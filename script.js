/* =====================================================
   PAGE NAVIGATION
===================================================== */

function showPage(pageId, button) {

    // Hide every page
    const pages = document.querySelectorAll(".page");

    pages.forEach(function(page) {

        page.classList.remove("active-page");

    });


    // Show selected page
    const selectedPage =
        document.getElementById(pageId);

    if (selectedPage) {

        selectedPage.classList.add("active-page");

    }


    // Remove active navigation
    const navItems =
        document.querySelectorAll(".nav-item");

    navItems.forEach(function(item) {

        item.classList.remove("active");

    });


    // Activate selected navigation
    if (button) {

        button.classList.add("active");

    }


    // Scroll to top
    window.scrollTo({

        top: 0,

        behavior: "smooth"

    });

}


/* =====================================================
   OPEN PREDICTION PAGE
===================================================== */

function openPrediction() {

    const buttons =
        document.querySelectorAll(".nav-item");

    showPage(
        "prediction",
        buttons[1]
    );

}


/* =====================================================
   DEMO PREDICTION
===================================================== */

function predictYield() {

    const crop =
        document.getElementById("crop").value;

    const state =
        document.getElementById("state").value;

    const costA2FL =
        parseFloat(
            document.getElementById("costA2FL").value
        );

    const costC2 =
        parseFloat(
            document.getElementById("costC2").value
        );

    const productionCost =
        parseFloat(
            document.getElementById("productionCost").value
        );


    if (
        isNaN(costA2FL) ||
        isNaN(costC2) ||
        isNaN(productionCost)
    ) {

        alert(
            "Please enter all cost values."
        );

        return;

    }


    /*
       IMPORTANT:

       This is currently a FRONTEND DEMO.

       Your actual Random Forest model is a
       Python .pkl file.

       Later we will connect this function
       to Flask/FastAPI so the real ML model
       generates the prediction.
    */


    let prediction = 15.36;


    // Demo values for visual testing

    if (crop === "SUGARCANE") {

        prediction = 937.38;

    }

    else if (crop === "COTTON") {

        prediction = 18.77;

    }

    else if (crop === "MAIZE") {

        prediction = 30.80;

    }

    else if (crop === "WHEAT") {

        prediction = 15.36;

    }

    else if (crop === "PADDY") {

        prediction = 44.20;

    }


    // Display prediction

    document.getElementById(
        "predictionValue"
    ).textContent =
        prediction.toFixed(2);


    document.getElementById(
        "predictionMessage"
    ).innerHTML =

        `The model estimates approximately
        <strong>${prediction.toFixed(2)}
        quintal/hectare</strong>
        for <strong>${crop}</strong>
        in <strong>${state}</strong>.`;

}