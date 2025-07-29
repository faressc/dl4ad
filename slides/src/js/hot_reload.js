// Hot reload WebSocket connection
(function() {
  const ws = new WebSocket(`ws://${window.location.host}`);
  
  ws.onmessage = function(event) {
    if (event.data === 'reload') {
      window.location.reload();
    }
  };
  
  ws.onopen = function() {
    console.log('🔥 Hot reload connected');
  };
  
  ws.onclose = function() {
    console.log('🔌 Hot reload disconnected');
  };
})();
