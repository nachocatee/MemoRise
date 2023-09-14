import { View, StyleSheet, Image } from "react-native";

import { calculateDynamicWidth } from "../constants/dynamicSize";

const ModifyInfoScreen = () => {
  return (
    <View style={styles.container}>
      <Image
        style={[styles.circle]}
        source={require("../assets/image/profile_bg.png")}
      />
    </View>
  );
};

export default ModifyInfoScreen;

const styles = StyleSheet.create({
  container: {
    alignItems: "center",
  },
  circle: {
    width: calculateDynamicWidth(103),
    height: calculateDynamicWidth(103),
  },
});
