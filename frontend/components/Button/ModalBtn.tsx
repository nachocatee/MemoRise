import React from "react";
import {
  Text,
  StyleSheet,
  Pressable,
  GestureResponderEvent,
} from "react-native";

import Colors from "../../constants/colors";
import { calculateDynamicWidth } from "../../constants/dynamicSize";

type ModalBtnProps = {
  children: React.ReactNode;
  onPress: (event: GestureResponderEvent) => void;
  color: string;
};

const ModalBtn: React.FC<ModalBtnProps> = ({ children, onPress, color }) => {
  return (
    <Pressable
      style={({ pressed }) =>
        pressed
          ? [styles.buttonInnerContainer, styles.pressed]
          : [styles.buttonInnerContainer, { backgroundColor: color }]
      }
      onPress={onPress}
    >
      <Text style={styles.buttonText}>{children}</Text>
    </Pressable>
  );
};

export default ModalBtn;

const styles = StyleSheet.create({
  buttonInnerContainer: {
    // backgroundColor: Colors.primary200,
    width: calculateDynamicWidth(75),
    height: calculateDynamicWidth(26),
    borderRadius: calculateDynamicWidth(30),
    justifyContent: "center",
    alignItems: "center",
    elevation: 4,
  },
  buttonText: {
    color: Colors.text,
    fontFamily: "Pretendard-Regular",
    fontSize: calculateDynamicWidth(16),
    letterSpacing: -0.5,
  },
  pressed: {
    backgroundColor: Colors.hover,
  },
});
