import { View, Text, StyleSheet, Image, Pressable } from "react-native";
import LinearGradient from "react-native-linear-gradient";

import { calculateDynamicWidth } from "../../constants/dynamicSize";
import Colors from "../../constants/colors";
import CancelHeader from "../../components/Header/CancelHeader";
import ProfilePic from "../../components/ProfilePic";

const MenuScreen = () => {
  return (
    <LinearGradient
      // colors={["#F5F5F5", "red"]}
      colors={["#F5F5F5", "#E9E9E9"]}
      start={{ x: 0, y: 0 }}
      end={{ x: 0, y: 1 }}
      style={{ flex: 1 }}
    >
      <CancelHeader />
      <View style={styles.container}>
        <View style={{ marginBottom: calculateDynamicWidth(4) }}>
          <ProfilePic />
        </View>
        {/* 유저 닉네임 */}
        <Text style={styles.text}>권소정</Text>
        {/* 유저 이메일 */}
        <View style={styles.emailBox}>
          <Image
            source={require("../../assets/image/kakao_sm.png")}
            style={styles.kakao}
          />
          <Text style={styles.email}>flfk33@naver.com</Text>
        </View>
        {/* 노트 이미지 */}
        <View style={{ marginTop: calculateDynamicWidth(20) }}>
          <Image
            source={require("../../assets/image/note.png")}
            style={styles.note}
          />
          {/* 메뉴 contents */}
          <View style={styles.menuList}>
            <Pressable
              style={[
                styles.menuItem,
                { marginBottom: calculateDynamicWidth(7) },
              ]}
            >
              <Image
                source={require("../../assets/icons/bookmark.png")}
                style={styles.bookmark}
              />
              <Text style={styles.text}>저장된 메모</Text>
            </Pressable>
            <Pressable style={styles.menuItem}>
              <View
                style={[
                  styles.dotIcon,
                  styles.menuIcon,
                  { backgroundColor: Colors.blue500 },
                ]}
              ></View>
              <Text style={styles.text}>전체 메모</Text>
            </Pressable>
            <Pressable style={styles.menuItem}>
              <View
                style={[
                  styles.dotIcon,
                  styles.menuIcon,
                  { backgroundColor: "#9CBDFF" },
                ]}
              ></View>
              <Text style={styles.text}>내 메모</Text>
            </Pressable>

            <View style={styles.line}></View>

            <Pressable style={styles.menuItem}>
              <View
                style={[
                  styles.dotIcon,
                  styles.menuIcon,
                  { backgroundColor: "#DAA418" },
                ]}
              ></View>
              <Text style={styles.text}>내 그룹</Text>
            </Pressable>
            <Pressable style={styles.menuItem}>
              <View
                style={[
                  styles.dotIcon,
                  styles.menuIcon,
                  { backgroundColor: Colors.primary300 },
                ]}
              ></View>
              <Text style={styles.text}>그룹 찾기</Text>
            </Pressable>

            <View style={styles.line}></View>

            <Pressable
              style={[styles.menuItem, { marginTop: calculateDynamicWidth(7) }]}
            >
              <Image
                source={require("../../assets/icons/setting.png")}
                style={styles.bookmark}
              />
              <Text style={styles.text}>회원정보 수정</Text>
            </Pressable>
          </View>
        </View>
      </View>
    </LinearGradient>
  );
};

export default MenuScreen;

const styles = StyleSheet.create({
  container: {
    flex: 1,
    alignItems: "center",
  },
  text: {
    fontSize: calculateDynamicWidth(18),
    fontFamily: "Pretendard-Medium",
    color: Colors.text,
  },
  emailBox: {
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "center",
    backgroundColor: Colors.hover,
    width: calculateDynamicWidth(155),
    borderRadius: calculateDynamicWidth(20) / 2,
    opacity: 0.5,
    marginTop: calculateDynamicWidth(8),
  },
  email: {
    fontSize: calculateDynamicWidth(15),
    color: Colors.text,
  },
  kakao: {
    width: calculateDynamicWidth(12),
    height: calculateDynamicWidth(12),
    marginRight: calculateDynamicWidth(5),
  },
  note: {
    width: calculateDynamicWidth(320),
    height: calculateDynamicWidth(438),
  },
  menuList: {
    position: "absolute",
    top: "18%",
    left: "11%",
  },
  menuItem: {
    flexDirection: "row",
    alignItems: "center",
  },
  menuIcon: {
    marginRight: calculateDynamicWidth(20),
    marginVertical: calculateDynamicWidth(15),
  },
  bookmark: {
    width: calculateDynamicWidth(15.52),
    height: calculateDynamicWidth(21),
    marginRight: calculateDynamicWidth(18),
  },
  dotIcon: {
    width: calculateDynamicWidth(12),
    height: calculateDynamicWidth(12),
    borderRadius: calculateDynamicWidth(12) / 2,
    marginLeft: calculateDynamicWidth(2),
  },
  line: {
    borderBottomWidth: 1,
    borderBottomColor: Colors.hover,
    height: calculateDynamicWidth(14),
    width: calculateDynamicWidth(245),
    marginBottom: calculateDynamicWidth(14),
  },
});
