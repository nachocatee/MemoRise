// 라이브러리
import { View, Text, StyleSheet, ActivityIndicator } from "react-native";

// 스타일
import Colors from "../constants/colors";

const LoadingOverlay = () => {
  return (
    <View style={styles.Container}>
      <ActivityIndicator size="large" color={Colors.text} />
    </View>
  );
};

export default LoadingOverlay;

const styles = StyleSheet.create({
  Container: {
    flex: 1,
    justifyContent: "center",
    alignItems: "center",
  },
});
