import { View, StyleSheet, Image } from "react-native";

const HighlightHeader = () => {
  return (
    <View style={styles.header}>
      <Image
        source={require("../../assets/image/logo/logo.png")}
        style={styles.logo}
      />
    </View>
  );
};

export default HighlightHeader;

const styles = StyleSheet.create({
  header: {
    height: 97,
    justifyContent: "center",
    alignItems: "center",
  },
  logo: {
    width: 189,
    height: 35,
  },
});
