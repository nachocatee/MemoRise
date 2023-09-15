import { View, StyleSheet, Image } from "react-native";

const CancelHeaderHighlight = () => {
  return (
    <View style={styles.header}>
      <Image
        source={require("../../assets/image/logo/logo.png")}
        style={styles.logo}
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
});
