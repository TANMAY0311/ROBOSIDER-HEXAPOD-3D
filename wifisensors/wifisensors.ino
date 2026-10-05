#include <WiFi.h>
#include <WebServer.h>
#include <DHT.h>

// =====================================================
// ROBO-DOG ESP32 SENSOR + WIFI DASHBOARD
// =====================================================

// -------------------------
// WiFi Access Point
// -------------------------

const char* AP_SSID = "ROBO-DOG";
const char* AP_PASSWORD = "robodog123";

WebServer server(80);


// =====================================================
// DHT11
// =====================================================

#define DHT_PIN 33
#define DHT_TYPE DHT11

DHT dht(DHT_PIN, DHT_TYPE);


// =====================================================
// MQ GAS SENSOR
// =====================================================

#define MQ_AO_PIN 34
#define MQ_DO_PIN 22


// =====================================================
// SENSOR VARIABLES
// =====================================================

float temperature = 0.0;
float humidity = 0.0;

int gasRaw = 0;
int gasDigital = HIGH;


// =====================================================
// READ SENSORS
// =====================================================

void readSensors() {

  float newTemperature = dht.readTemperature();
  float newHumidity = dht.readHumidity();

  if (!isnan(newTemperature)) {
    temperature = newTemperature;
  }

  if (!isnan(newHumidity)) {
    humidity = newHumidity;
  }

  gasRaw = analogRead(MQ_AO_PIN);

  gasDigital = digitalRead(MQ_DO_PIN);
}


// =====================================================
// SENSOR API
// =====================================================

void handleSensors() {

  readSensors();

  String json = "{";

  json += "\"temperature\":";
  json += String(temperature, 1);

  json += ",\"humidity\":";
  json += String(humidity, 1);

  json += ",\"gas_raw\":";
  json += String(gasRaw);

  json += ",\"gas_alert\":";

  if (gasDigital == LOW) {
    json += "true";
  } else {
    json += "false";
  }

  json += "}";

  server.send(
    200,
    "application/json",
    json
  );
}


// =====================================================
// MAIN DASHBOARD
// =====================================================

void handleRoot() {

  String html = R"rawliteral(

<!DOCTYPE html>

<html>

<head>

<meta charset="UTF-8">

<meta name="viewport"
      content="width=device-width, initial-scale=1.0">

<meta http-equiv="Content-Type"
      content="text/html; charset=UTF-8">

<title>RoboDog Environmental Dashboard</title>


<style>

/* -----------------------------------------
   PAGE
----------------------------------------- */

body {

  margin: 0;

  padding: 0;

  font-family:
    Arial,
    Helvetica,
    sans-serif;

  background: #0b1220;

  color: #ffffff;

}


/* -----------------------------------------
   HEADER
----------------------------------------- */

.header {

  padding: 25px 20px 20px 20px;

  text-align: center;

  background: #111827;

  border-bottom:
    1px solid #273449;

}


.header h1 {

  margin: 0;

  font-size: 30px;

  font-weight: 700;

}


.header p {

  margin-top: 8px;

  margin-bottom: 0;

  color: #9ca3af;

  font-size: 14px;

}


/* -----------------------------------------
   STATUS
----------------------------------------- */

.connection {

  display: inline-block;

  margin-top: 15px;

  padding: 7px 14px;

  border-radius: 20px;

  background: #123d2a;

  color: #22c55e;

  font-size: 13px;

  font-weight: bold;

}


/* -----------------------------------------
   CONTAINER
----------------------------------------- */

.container {

  max-width: 900px;

  margin: auto;

  padding: 20px;

}


/* -----------------------------------------
   SECTION TITLE
----------------------------------------- */

.section-title {

  margin-top: 10px;

  margin-bottom: 15px;

  color: #d1d5db;

  font-size: 18px;

  font-weight: bold;

}


/* -----------------------------------------
   SENSOR GRID
----------------------------------------- */

.grid {

  display: grid;

  grid-template-columns:
    repeat(2, 1fr);

  gap: 15px;

}


/* -----------------------------------------
   CARD
----------------------------------------- */

.card {

  background: #182334;

  border:
    1px solid #263449;

  border-radius: 18px;

  padding: 22px;

  text-align: center;

}


/* -----------------------------------------
   SENSOR NAME
----------------------------------------- */

.card-title {

  color: #9ca3af;

  font-size: 15px;

  margin-bottom: 10px;

}


/* -----------------------------------------
   SENSOR VALUE
----------------------------------------- */

.value {

  font-size: 34px;

  font-weight: bold;

}


/* -----------------------------------------
   UNIT
----------------------------------------- */

.unit {

  color: #9ca3af;

  font-size: 15px;

  margin-left: 4px;

}


/* -----------------------------------------
   GAS CARD
----------------------------------------- */

.gas-card {

  margin-top: 15px;

}


/* -----------------------------------------
   GAS STATUS
----------------------------------------- */

.status {

  margin-top: 10px;

  padding: 13px;

  border-radius: 12px;

  font-size: 18px;

  font-weight: bold;

}


.normal {

  background: #123d2a;

  color: #22c55e;

}


.alert {

  background: #4a1717;

  color: #ef4444;

}


/* -----------------------------------------
   FOOTER
----------------------------------------- */

.footer {

  text-align: center;

  color: #6b7280;

  font-size: 12px;

  padding: 30px 10px;

}


/* -----------------------------------------
   MOBILE
----------------------------------------- */

@media (max-width: 600px) {

  .grid {

    grid-template-columns:
      1fr;

  }

  .header h1 {

    font-size: 26px;

  }

}

</style>

</head>


<body>


<!-- =====================================
     HEADER
===================================== -->

<div class="header">

  <h1>RoboDog</h1>

  <p>Environmental Monitoring System</p>

  <div
    id="connection"
    class="connection">

    ESP32 CONNECTED

  </div>

</div>


<!-- =====================================
     MAIN
===================================== -->

<div class="container">


  <div class="section-title">

    Environmental Sensors

  </div>


  <!-- SENSOR GRID -->

  <div class="grid">


    <!-- TEMPERATURE -->

    <div class="card">

      <div class="card-title">
        Temperature
      </div>

      <div class="value">

        <span id="temperature">
          --
        </span>

        <span class="unit">
          deg C
        </span>

      </div>

    </div>


    <!-- HUMIDITY -->

    <div class="card">

      <div class="card-title">
        Humidity
      </div>

      <div class="value">

        <span id="humidity">
          --
        </span>

        <span class="unit">
          %
        </span>

      </div>

    </div>


  </div>


  <!-- =================================
       GAS SENSOR
  ================================== -->

  <div class="card gas-card">

    <div class="card-title">

      Gas Sensor

    </div>


    <div class="value">

      <span id="gas">
        --
      </span>

    </div>


    <div class="unit">

      Raw sensor value

    </div>


    <div
      id="gasStatus"
      class="status normal">

      GAS STATUS: NORMAL

    </div>

  </div>


</div>


<!-- =====================================
     FOOTER
===================================== -->

<div class="footer">

  RoboDog Sensor Node<br>

  ESP32 Environmental Monitoring

</div>


<!-- =====================================
     JAVASCRIPT
===================================== -->

<script>


async function updateSensors() {


  try {


    const response =
      await fetch(
        "/api/sensors"
      );


    const data =
      await response.json();


    // -----------------------------
    // Temperature
    // -----------------------------

    document
      .getElementById(
        "temperature"
      )
      .textContent =
        data.temperature.toFixed(1);


    // -----------------------------
    // Humidity
    // -----------------------------

    document
      .getElementById(
        "humidity"
      )
      .textContent =
        data.humidity.toFixed(1);


    // -----------------------------
    // Gas
    // -----------------------------

    document
      .getElementById(
        "gas"
      )
      .textContent =
        data.gas_raw;


    // -----------------------------
    // Gas Status
    // -----------------------------

    const status =
      document.getElementById(
        "gasStatus"
      );


    if (data.gas_alert) {


      status.textContent =
        "GAS ALERT: THRESHOLD EXCEEDED";


      status.className =
        "status alert";


    }

    else {


      status.textContent =
        "GAS STATUS: NORMAL";


      status.className =
        "status normal";


    }


    // -----------------------------
    // Connection status
    // -----------------------------

    const connection =
      document.getElementById(
        "connection"
      );


    connection.textContent =
      "ESP32 CONNECTED";


    connection.style.color =
      "#22c55e";


  }


  catch(error) {


    const connection =
      document.getElementById(
        "connection"
      );


    connection.textContent =
      "ESP32 DISCONNECTED";


    connection.style.color =
      "#ef4444";


  }


}


// Initial update

updateSensors();


// Update every 2 seconds

setInterval(
  updateSensors,
  2000
);


</script>


</body>

</html>

)rawliteral";


  // Explicit UTF-8 response

  server.send(
    200,
    "text/html; charset=UTF-8",
    html
  );
}


// =====================================================
// SETUP
// =====================================================

void setup() {

  Serial.begin(115200);

  delay(1000);


  // DHT11

  dht.begin();


  // MQ sensor

  pinMode(
    MQ_AO_PIN,
    INPUT
  );

  pinMode(
    MQ_DO_PIN,
    INPUT
  );


  analogReadResolution(12);


  // ===================================================
  // START WIFI ACCESS POINT
  // ===================================================

  WiFi.mode(WIFI_AP);

  WiFi.softAP(
    AP_SSID,
    AP_PASSWORD
  );


  // ===================================================
  // SERIAL INFORMATION
  // ===================================================

  Serial.println();

  Serial.println(
    "=========================================="
  );

  Serial.println(
    "        ROBODOG SENSOR SERVER"
  );

  Serial.println(
    "=========================================="
  );

  Serial.println();

  Serial.print(
    "WiFi Name : "
  );

  Serial.println(
    AP_SSID
  );


  Serial.print(
    "Password  : "
  );

  Serial.println(
    AP_PASSWORD
  );


  Serial.print(
    "IP Address: "
  );

  Serial.println(
    WiFi.softAPIP()
  );


  Serial.println();


  // ===================================================
  // SERVER ROUTES
  // ===================================================

  server.on(
    "/",
    handleRoot
  );


  server.on(
    "/api/sensors",
    handleSensors
  );


  server.begin();


  Serial.println(
    "Web server started."
  );

  Serial.println(
    "=========================================="
  );

}


// =====================================================
// LOOP
// =====================================================

void loop() {

  server.handleClient();

}