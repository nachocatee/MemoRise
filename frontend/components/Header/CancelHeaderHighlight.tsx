import { View, StyleSheet, Image } from "react-native";

const CancelHeaderHighlight = () => {
  return (
    <View style={styles.header}>
      <Image
        source={require("../../assets/image/logo/logo.png")}
        style={styles.logo}
      />
      <Image
        source={require("../../assets/icons/cancel.png")}
        style={styles.cancel}
      />
    </View>
  );
};

export default CancelHeaderHighlight;

const styles = StyleSheet.create({
  header: {
    height: 97,
    justifyContent: "center",
    alignItems: "center",
    flexDirection: "row",
  },
  logo: {
    width: 189,
    height: 33,
  },
  cancel: {
    width: 22.92,
    height: 22.92,
    position: "absolute",
    right: "7%",
  },
});
