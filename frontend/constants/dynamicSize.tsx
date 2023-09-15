import { Dimensions } from "react-native";

type DynamicWidthFunction = (width: number) => number;

export const calculateDynamicWidth: DynamicWidthFunction = (width) => {
  const screenWidth = Dimensions.get("window").width;
  const baseScreenWidth = 360; // 기준 너비

  const dynamicWidth = (screenWidth / baseScreenWidth) * width;

  return dynamicWidth;
};
