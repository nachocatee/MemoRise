import { useNavigation } from "@react-navigation/native";
import { StackNavigationProp } from "@react-navigation/stack";
import { View, StyleSheet, Image, Pressable } from "react-native";

type RootStackParamList = {
  Main: undefined;
};

const CancelHeader = () => {
  const navigation =
    useNavigation<StackNavigationProp<RootStackParamList, "Main">>();

  return (
    <View style={styles.header}>
      <Pressable onPress={() => navigation.navigate("Main")}>
        <Image
          source={require("../../assets/image/logo/logoblack.png")}
          style={styles.logo}
        />
      </Pressable>
      <Pressable
        onPress={() => navigation.navigate("Main")}
        style={styles.cancelContainer}
      >
        <Image
          source={require("../../assets/icons/cancel.png")}
          style={styles.cancel}
        />
      </Pressable>
    </View>
  );
};

export default CancelHeader;

const styles = StyleSheet.create({
  header: {
    height: 97,
    justifyContent: "center",
    alignItems: "center",
    flexDirection: "row",
  },
  logo: {
    width: 189,
    height: 35,
  },
  cancelContainer: {
    position: "absolute",
    right: 25,
  },
  cancel: {
    width: 22.92,
    height: 22.92,
  },
});
