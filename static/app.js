async function checkHealth() {
  try {
    const r = await fetch("/dualing-simulation/api/health");
    const j = await r.json();
    document.getElementById("health").textContent = JSON.stringify(j);
  } catch (e) {
    document.getElementById("health").textContent = "error: " + e.message;
  }
}
checkHealth();
