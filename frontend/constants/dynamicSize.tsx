import { Dimensions } from "react-native";

type DynamicWidthFunction = (width: number) => number;
type DynamicHeightFunction = (
  width: number,
  height: number,
  dynamicWidth: number
) => number;

export const calculateDynamicWidth: DynamicWidthFunction = (width) => {
  const screenWidth = Dimensions.get("window").width;
  const baseScreenWidth = 360; // 기준 너비

  const dynamicWidth = (screenWidth / baseScreenWidth) * width;

  return dynamicWidth;
};

export const calculateDynamicHeight: DynamicHeightFunction = (
  width,
  height,
  dynamicWidth
) => {
  const aspectRatio = width / height;

  const dynamicHeight = dynamicWidth / aspectRatio;

  return dynamicHeight;
};
