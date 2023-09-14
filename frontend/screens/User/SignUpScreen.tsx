import { View, Text, StyleSheet, Image, TextInput } from "react-native";
import LinearGradient from "react-native-linear-gradient";

import CancelHeaderHighlight from "../../components/Header/CancelHeaderHighlight";
import ProfilePic from "../../components/ProfilePic";
import Colors from "../../constants/colors";
import { calculateDynamicWidth } from "../../constants/dynamicSize";

const SignUpScreen = () => {
  return (
    <LinearGradient
      // colors={["#F5F5F5", "red"]}
      colors={["#F5F5F5", "#E9E9E9"]}
      start={{ x: 0, y: 0 }}
      end={{ x: 0, y: 1 }}
      style={{ flex: 1 }}
    >
      <CancelHeaderHighlight />
      <View style={styles.container}>
        <View style={styles.profilebox}>
          <ProfilePic />
          <Image
            source={require("../../assets/icons/album.png")}
            style={styles.album}
          />
        </View>
        <View style={styles.inputBox}>
          <Text style={styles.text}>닉네임</Text>
          <TextInput />
        </View>
      </View>
    </LinearGradient>
  );
};

export default SignUpScreen;

const styles = StyleSheet.create({
  container: {
    flex: 1,
    alignItems: "center",
  },
  text: {
    fontSize: calculateDynamicWidth(20),
    fontFamily: "Pretendard-SemiBold",
    color: Colors.text,
  },
  profilebox: {
    width: calculateDynamicWidth(103),
  },
  album: {
    width: calculateDynamicWidth(31),
    height: calculateDynamicWidth(31),
    position: "absolute",
    bottom: 0,
    right: 0,
  },
  inputBox: {
    width: calculateDynamicWidth(285),
  },
});
