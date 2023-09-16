import { useNavigation } from "@react-navigation/native";
import { StackNavigationProp } from "@react-navigation/stack";
import { View, StyleSheet, Image, Pressable } from "react-native";

type RootStackParamList = {
  Menu: undefined;
};

const MainHeader = () => {
  const navigation =
    useNavigation<StackNavigationProp<RootStackParamList, "Menu">>();

  return (
    <View style={styles.header}>
      <Image
        source={require("../../assets/icons/notification_none.png")}
        style={styles.notification}
      />
      <Image
        source={require("../../assets/image/logo/logowhite.png")}
        style={styles.logo}
      />
      <Pressable
        style={styles.menu}
        onPress={() => navigation.navigate("Menu")}
      ></Pressable>
    </View>
  );
};

export default MainHeader;

const styles = StyleSheet.create({
  header: {
    height: 97,
    justifyContent: "space-between",
    alignItems: "center",
    flexDirection: "row",
    paddingHorizontal: 20,
  },
  notification: {
    width: 26.25,
    height: 28.43,
  },
  logo: {
    width: 189,
    height: 35,
  },
  menu: {
    width: 31,
    height: 15,
    borderTopWidth: 2,
    borderTopColor: "white",
    borderBottomWidth: 2,
    borderBottomColor: "white",
  },
});
