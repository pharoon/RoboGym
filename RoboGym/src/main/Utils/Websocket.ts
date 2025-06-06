import WebSocket from 'ws';

class WebSocketConnector{
	private ws:WebSocket;
    constructor(url: string){
      try {
        this.ws = new WebSocket(url)

        this.ws.on('open', ()=> {
          console.log('Connected to the WebSocket server!');
          const interva = setInterval(()=>{
              this.ws?.send(JSON.stringify({ cmd:"step" }));
          },2000)
          
      });
  
      // Listen for messages from the WebSocket server (PyBullet)
      this.ws.on('message', function incoming(data) {
          console.log("--------------------------------------------------------------\n")
          console.log('Received from PyBullet:', JSON.parse(data));
          
      });
  
      this.ws.on('close', () => {
          console.log('WebSocket connection closed');
      });
  
      this.ws.on('error', (error) => {
          console.error('WebSocket Error:', error);
      });  
      } catch (error) {
          console.log("Websocket error occurred", error)
      }
  }


}

export { WebSocketConnector }