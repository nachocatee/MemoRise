import React, { useState } from "react";
import { Button } from "react-native";
import { RTCPeerConnection, RTCView, mediaDevices } from "react-native-webrtc";

const Test: React.FC = () => {
  const [localStream, setLocalStream] = useState(null);
  const [remoteStream, setRemoteStream] = useState(null);
  let pc: any;

  const start = async () => {
    const configuration = {
      iceServers: [{ urls: "stun:stun.l.google.com:19302" }],
    };
    pc = new RTCPeerConnection(configuration);

    pc.onicecandidate = (event: any) => {
      if (event.candidate) {
        // Send this candidate to the remote peer (via your signaling server)
      }
    };

    pc.onaddstream = (event: any) => {
      setRemoteStream(event.stream);
    };

    const constraints = {
      audio: true,
      video: {
        mandatory: {
          minWidth: 500,
          minHeight: 300,
          minFrameRate: 30,
        },
        facingMode: "user",
        optional: [{ sourceId: "your-source-id" }],
      },
    };

    const stream = await mediaDevices.getUserMedia(constraints);
    setLocalStream(stream);

    stream.getTracks().forEach((track) => {
      pc.addTrack(track, stream);
    });

    // Here, you'd also set up your code to create an offer and send it to the remote peer.
    // And handle the answer when it comes back.
    // This is similar to the `negotiate` function in your original code.
  };

  const stop = () => {
    if (pc) {
      pc.close();
    }
    // Also stop local streams
    if (localStream) {
      localStream.getTracks().forEach((track: any) => {
        track.stop();
      });
      setLocalStream(null);
    }
  };

  return (
    <>
      {localStream && <RTCView streamURL={(localStream as any).toURL()} />}
      {remoteStream && <RTCView streamURL={(remoteStream as any).toURL()} />}
      <Button title="Start" onPress={start} />
      <Button title="Stop" onPress={stop} />
    </>
  );
};

export default Test;
