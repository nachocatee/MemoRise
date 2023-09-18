import React from "react";
import { View, Image, TextInput, StyleSheet, Pressable } from "react-native";

import Colors from "../constants/colors";
import { calculateDynamicWidth } from "../constants/dynamicSize";

const SeachInput = () => {
  return (
    <View style={styles.inputContainer}>
      <TextInput style={styles.input} />
      <Pressable>
        <Image
          source={require("../assets/icons/search.png")}
          style={styles.image}
        />
      </Pressable>
    </View>
  );
};

export default SeachInput;

const styles = StyleSheet.create({
  inputContainer: {
    width: calculateDynamicWidth(306),
    height: calculateDynamicWidth(44),
    backgroundColor: "rgba(255, 255, 255, 0.7)",
    borderRadius: calculateDynamicWidth(15),
    paddingHorizontal: calculateDynamicWidth(15),
    flexDirection: "row",
    justifyContent: "space-between",
    alignItems: "center",
  },
  input: {
    color: Colors.text,
    fontFamily: "Pretendard-Regular",
    fontSize: calculateDynamicWidth(17),
  },
  image: {
    // position: "absolute",
    width: calculateDynamicWidth(17),
    height: calculateDynamicWidth(17),
  },
});
