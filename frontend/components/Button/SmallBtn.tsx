import React from "react";
import {
  Text,
  StyleSheet,
  Pressable,
  GestureResponderEvent,
} from "react-native";

import Colors from "../../constants/colors";
import { calculateDynamicWidth } from "../../constants/dynamicSize";

type SmallBtnProps = {
  children: React.ReactNode;
  onPress: (event: GestureResponderEvent) => void;
};

const SmallBtn: React.FC<SmallBtnProps> = ({ children, onPress }) => {
  return (
    <Pressable
      style={({ pressed }) =>
        pressed
          ? [styles.buttonInnerContainer, styles.pressed]
          : styles.buttonInnerContainer
      }
      onPress={onPress}
    >
      <Text style={styles.buttonText}>{children}</Text>
    </Pressable>
  );
};

export default SmallBtn;

const styles = StyleSheet.create({
  buttonInnerContainer: {
    backgroundColor: Colors.primary200,
    width: calculateDynamicWidth(66),
    height: calculateDynamicWidth(30),
    borderRadius: calculateDynamicWidth(10),
    justifyContent: "center",
    alignItems: "center",
    elevation: 4,
  },
  buttonText: {
    color: Colors.text,
    fontFamily: "Pretendard-Regular",
    fontSize: calculateDynamicWidth(18),
    letterSpacing: -0.5,
  },
  pressed: {
    backgroundColor: Colors.hover,
  },
});
