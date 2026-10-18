(function () {
  "use strict";

  var $ = function (id) { return document.getElementById(id); };
  var form = $("crop-form");
  var errorBox = $("form-error");
  var submitBtn = $("submit-btn");
  var keys = FIELD_DEFAULTS.map(function (f) { return f.key; });

  function setVal(key, v) {
    [key, key + "-range"].forEach(function (id) {
      var el = $(id);
      if (el) { el.value = v; }
    });
  }

  // slider <-> number box sync
  keys.forEach(function (key) {
    var num = $(key), rng = $(key + "-range");
    rng.addEventListener("input", function () { num.value = rng.value; });
    num.addEventListener("input", function () { rng.value = num.value; });
  });

  // sample farms
  var chips = document.querySelectorAll(".chip");
  chips.forEach(function (chip) {
    chip.addEventListener("click", function () {
      var data = JSON.parse(chip.getAttribute("data-preset"));
      Object.keys(data).forEach(function (k) { setVal(k, data[k]); });
      chips.forEach(function (c) { c.classList.remove("on"); });
      chip.classList.add("on");
      hideError();
    });
  });

  // reset
  $("reset-btn").addEventListener("click", function () {
    FIELD_DEFAULTS.forEach(function (f) { setVal(f.key, f.default); });
    chips.forEach(function (c) { c.classList.remove("on"); });
    hideError();
    $("result").hidden = true;
  });

  function showError(msg) { errorBox.textContent = msg; errorBox.hidden = false; }
  function hideError() { errorBox.hidden = true; }
  function setLoading(on) {
    submitBtn.classList.toggle("loading", on);
    submitBtn.disabled = on;
  }

  // predict
  form.addEventListener("submit", async function (e) {
    e.preventDefault();
    hideError();
    var payload = {};
    keys.forEach(function (k) { payload[k] = $(k).value; });
    setLoading(true);
    try {
      var res = await fetch("/api/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload)
      });
      var data = await res.json();
      if (!res.ok) { showError(data.error || "Something went wrong."); return; }
      render(data);
    } catch (err) {
      showError("Could not reach the server. Is app.py running?");
    } finally {
      setLoading(false);
    }
  });

  function render(d) {
    $("result-name").textContent = d.name;
    $("result-confidence").textContent = d.confidence;
    $("result-ring").style.setProperty("--p", d.confidence);
    $("result-description").textContent = d.description;
    $("result-fertilizer").textContent = d.fertilizer;
    $("result-season").textContent = d.season;
    $("result-soil").textContent = d.soil;

    var photo = $("result-photo");
    photo.innerHTML = "";
    if (d.image) {
      var img = document.createElement("img");
      img.src = d.image;
      img.alt = d.name;
      photo.appendChild(img);
    } else {
      photo.textContent = d.emoji;
    }

    var bars = $("result-bars");
    bars.innerHTML = "";
    d.alternatives.forEach(function (a) {
      var li = document.createElement("li");
      var name = document.createElement("span");
      name.textContent = a.emoji + " " + a.name;
      var track = document.createElement("span");
      track.className = "track";
      var fill = document.createElement("i");
      fill.style.width = a.percent + "%";
      track.appendChild(fill);
      var pct = document.createElement("em");
      pct.textContent = a.percent + "%";
      li.appendChild(name);
      li.appendChild(track);
      li.appendChild(pct);
      bars.appendChild(li);
    });

    var result = $("result");
    result.hidden = false;
    result.scrollIntoView({ behavior: "smooth", block: "start" });
  }

  $("print-btn").addEventListener("click", function () { window.print(); });

  // live weather
  $("weather-btn").addEventListener("click", async function () {
    var msg = $("weather-msg");
    var city = $("city").value.trim();
    msg.textContent = "Loading...";
    try {
      var res = await fetch("/api/weather?city=" + encodeURIComponent(city));
      var d = await res.json();
      if (!res.ok) { msg.textContent = d.error; return; }
      setVal("temperature", d.temperature);
      setVal("humidity", d.humidity);
      msg.textContent = d.city + ": " + d.temperature + " °C, humidity " + d.humidity + "%";
    } catch (e) {
      msg.textContent = "Could not fetch weather.";
    }
  });
})();
