import React from "react";
import { View, Text, StyleSheet } from "react-native";
import LinearGradient from "react-native-linear-gradient";

import Colors from "../constants/colors";
import { calculateDynamicWidth } from "../constants/dynamicSize";

type GroupBoxProps = {
  teamName: string;
};

const GroupBox: React.FC<GroupBoxProps> = ({ teamName }) => {
  return (
    <View>
      <LinearGradient
        colors={["#F5F5F5", Colors.hover]}
        start={{ x: 0, y: 0 }}
        end={{ x: 0, y: 1 }}
        style={styles.groupBox}
      ></LinearGradient>
      <Text style={styles.title}>{teamName}</Text>
    </View>
  );
};

export default GroupBox;

const styles = StyleSheet.create({
  groupBox: {
    width: calculateDynamicWidth(128),
    height: calculateDynamicWidth(128),
    borderRadius: calculateDynamicWidth(15),
  },
  title: {
    color: Colors.text,
    fontFamily: "Pretendard-Regular",
    fontSize: calculateDynamicWidth(18),
    textAlign: "center",
    marginTop: calculateDynamicWidth(5),
    marginBottom: calculateDynamicWidth(30),
  },
});
