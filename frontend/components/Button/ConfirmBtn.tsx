import React from "react";
import {
  Text,
  StyleSheet,
  Pressable,
  GestureResponderEvent,
} from "react-native";
import Colors from "../../constants/colors";
import { calculateDynamicWidth } from "../../constants/dynamicSize";

type ConfirmBtnProps = {
  children: React.ReactNode;
  onPress: (event: GestureResponderEvent) => void;
};

const ConfirmBtn: React.FC<ConfirmBtnProps> = ({ children, onPress }) => {
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

export default ConfirmBtn;

const styles = StyleSheet.create({
  buttonInnerContainer: {
    backgroundColor: Colors.primary200,
    width: calculateDynamicWidth(270),
    height: calculateDynamicWidth(66),
    borderRadius: calculateDynamicWidth(33),
    justifyContent: "center",
    alignItems: "center",
    elevation: 4,
  },
  buttonText: {
    color: Colors.text,
    fontFamily: "Pretendard-Regular",
    fontSize: calculateDynamicWidth(24),
    letterSpacing: -0.5,
  },
  pressed: {
    backgroundColor: Colors.hover,
  },
});
