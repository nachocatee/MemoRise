import { Dimensions } from "react-native";

type DynamicWidthFunction = (size: number) => number;

export const calculateDynamicWidth: DynamicWidthFunction = (size) => {
  const screenHeight = Dimensions.get("window").height;

  const baseScreenHeight = 800; // 기준 높이

  const dynamicSize = (screenHeight / baseScreenHeight) * size;

  return dynamicSize;
};
