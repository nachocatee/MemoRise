import { Dimensions } from "react-native";

type DynamicWidthFunction = (size: number) => number;

export const calculateDynamicWidth: DynamicWidthFunction = (size) => {
  // const screenWidth = Dimensions.get("window").width;
  const screenHeight = Dimensions.get("window").height;

  // const baseScreenWidth = 360; // 기준 너비
  const baseScreenHeight = 800; // 기준 높이

  // const dynamicWidth = (screenWidth / baseScreenWidth) * width;
  const dynamicSize = (screenHeight / baseScreenHeight) * size;

  return dynamicSize;
};
