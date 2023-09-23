package com.tjjhtjh.memorise.domain.user.controller;

import com.tjjhtjh.memorise.domain.team.service.TeamService;
import com.tjjhtjh.memorise.domain.team.service.dto.response.MyTeamListResponse;
import com.tjjhtjh.memorise.domain.user.repository.entity.User;
import com.tjjhtjh.memorise.domain.user.service.UserService;
import com.tjjhtjh.memorise.domain.user.service.dto.request.JoinRequest;
import com.tjjhtjh.memorise.domain.user.service.dto.request.UpdateUserInfoRequest;
import com.tjjhtjh.memorise.domain.user.service.dto.response.JoinResponse;
import com.tjjhtjh.memorise.domain.user.service.dto.response.UpdateUserInfoResponse;
import com.tjjhtjh.memorise.domain.user.service.dto.response.UserInfoResponse;
import com.tjjhtjh.memorise.domain.user.service.dto.response.UserListResponse;
import com.tjjhtjh.memorise.global.file.service.AwsS3Service;
import com.tjjhtjh.memorise.global.file.service.dto.CreateFileRequest;
import lombok.RequiredArgsConstructor;
import org.springframework.data.repository.query.Param;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.multipart.MultipartFile;

import java.util.List;

@RestController
@RequiredArgsConstructor
@RequestMapping("/user")
public class UserController {

    private final UserService userService;
    private final TeamService teamService;
    private final AwsS3Service awsS3Service;

    private static String dirName = "profile-image";

    @PostMapping("/upload")
    public ResponseEntity<CreateFileRequest> uploadMultipleFile(@RequestPart MultipartFile file) {
        return ResponseEntity.ok(awsS3Service.uploadMultiFile(file, dirName));
    }

    @PostMapping
    public ResponseEntity<JoinResponse> join(@RequestBody JoinRequest joinRequest) {
        User user = userService.join(joinRequest);
        return ResponseEntity.ok(new JoinResponse(true, user.getUserSeq()));
    }

    @PutMapping("/{userSeq}")
    public ResponseEntity<UpdateUserInfoResponse> update(@PathVariable Long userSeq, @RequestBody UpdateUserInfoRequest updateUserInfoRequest) {
        userService.updateUserInfo(userSeq, updateUserInfoRequest);
        return ResponseEntity.ok(new UpdateUserInfoResponse(true));
    }

    @GetMapping("/{userSeq}")
    public ResponseEntity<UserInfoResponse> getUserInfo(@PathVariable Long userSeq) {
        return ResponseEntity.ok(new UserInfoResponse(userService.getUserInfo(userSeq)));
    }

    @GetMapping("/list")
    public ResponseEntity<UserListResponse> getUserList(@RequestParam(required = false) String keyword) {
        return ResponseEntity.ok(userService.getUserList(keyword));
    }

    @GetMapping("/{userSeq}/my-teams")
    public ResponseEntity<List<MyTeamListResponse>> getMyTeamList(@PathVariable Long userSeq) {
        return ResponseEntity.ok(teamService.getMyTeamList(userSeq));
    }
}
