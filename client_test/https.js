import React, { useState, useRef } from 'react';
import { Button, Alert } from 'react-native';
import {
  RTCPeerConnection,
  RTCIceCandidate,
  RTCSessionDescription,
  mediaDevices,
} from 'react-native-webrtc';
import RNFetchBlob from 'rn-fetch-blob'; // <--- Import RNFetchBlob

const App = () => {
  const [isConnected, setIsConnected] = useState(false);
  const pc = useRef(null);

  const negotiate = async () => {
    try {
      await pc.current.addTransceiver('video', { direction: 'sendonly' });
      const offer = await pc.current.createOffer();
      await pc.current.setLocalDescription(offer);
      console.log(offer);

      // Use RNFetchBlob for the request
      const response = await RNFetchBlob.config({
        trusty: true // <-- SSL/TLS 인증서 검증 생략
    }).fetch('POST', 'https://70.12.130.111:8081//offer', {
        'Content-Type': 'application/json',
    }, JSON.stringify({
        sdp: offer.sdp,
        type: offer.type,
    }));

      const answer = await response.json();
      await pc.current.setRemoteDescription(new RTCSessionDescription(answer));
    } catch (e) {
      Alert.alert('Error', e.message);
    }
  };

  const start = async () => {
    const configuration = {
      sdpSemantics: 'unified-plan',
      iceServers: [{ urls: 'stun:stun.l.google.com:19302' }],
    };
    pc.current = new RTCPeerConnection(configuration);

    try {
      const stream = await mediaDevices.getUserMedia({
        audio: true,
        video: {
          width: 1280,
          height: 720,
          frameRate: 60,
          facingMode: 'environment',
        },
      });
      stream.getTracks().forEach(track => {
        pc.current.addTrack(track, stream);
      });

      await negotiate();
      setIsConnected(true);
    } catch (e) {
      Alert.alert('Error', 'Error starting the connection: ' + e.message);
    }
  };

  const stop = () => {
    setIsConnected(false);
    if (pc.current) {
      pc.current.close();
    }
  };

  return (
    <>
      <Button title="Start" onPress={start} disabled={isConnected} />
      <Button title="Stop" onPress={stop} disabled={!isConnected} />
    </>
  );
};

export default App;
