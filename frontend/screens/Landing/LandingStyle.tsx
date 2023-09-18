import { StyleSheet } from "react-native";

import { calculateDynamicWidth } from "../../constants/dynamicSize";

export const styles = StyleSheet.create({
  rootContainer: {
    flex: 1,
    justifyContent: "center",
    alignItems: "center",
  },
  image: {
    width: calculateDynamicWidth(312),
    height: calculateDynamicWidth(57),
  },
});
