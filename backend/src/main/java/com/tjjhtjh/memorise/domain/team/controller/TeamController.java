package com.tjjhtjh.memorise.domain.team.controller;

import com.tjjhtjh.memorise.domain.team.service.TeamService;
import com.tjjhtjh.memorise.domain.team.service.dto.request.*;
import com.tjjhtjh.memorise.domain.team.service.dto.response.*;
import lombok.RequiredArgsConstructor;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import java.util.List;

@RestController
@RequiredArgsConstructor
@RequestMapping("/teams")
public class TeamController {

    private final TeamService teamService;

    @PostMapping
    public ResponseEntity<CreateTeamResponse> createGroup(@RequestBody CreateTeamRequest createTeamRequest) {
        return ResponseEntity.ok(teamService.createTeam(createTeamRequest));
    }

    @GetMapping("/{userSeq}")
    public ResponseEntity<List<TeamListResponse>> getTeamList(@PathVariable Long userSeq, @RequestParam(required = false) String keyword) {
        return ResponseEntity.ok(teamService.getTeamList(userSeq, keyword));
    }

    @GetMapping("/{teamSeq}/{userSeq}")
    public ResponseEntity<TeamDetailResponse> getTeamInfo(@PathVariable Long teamSeq, @PathVariable Long userSeq) {
        return ResponseEntity.ok(teamService.getTeamDetailInfo(teamSeq, userSeq));
    }

    @PutMapping("/{teamSeq}")
    public ResponseEntity<UpdateTeamResponse> updateGroup(@PathVariable Long teamSeq, @RequestBody UpdateTeamRequest updateTeamRequest) {
        return ResponseEntity.ok(teamService.updateTeam(teamSeq, updateTeamRequest));
    }

    @PutMapping("/{teamSeq}/invite")
    public ResponseEntity<InviteMemberResponse> inviteMember(@PathVariable Long teamSeq, @RequestBody InviteMemberRequest inviteMemberRequest) {
        return ResponseEntity.ok(teamService.inviteMember(teamSeq, inviteMemberRequest));
    }

    @GetMapping("/{teamSeq}/invite/{userSeq}")
    public ResponseEntity<List<InviteUserListResponse>> getInviteUserList(@PathVariable Long teamSeq, @PathVariable Long userSeq, @RequestParam(required = false) String keyword) {
        return ResponseEntity.ok(teamService.getInviteUserList(teamSeq, userSeq, keyword));
    }

    @DeleteMapping("/{teamSeq}/kick")
    public ResponseEntity<KickMemberResponse> kickMember(@PathVariable Long teamSeq, @RequestBody KickMemberRequest kickMemberRequest) {
        teamService.kickMember(teamSeq, kickMemberRequest);
        return ResponseEntity.ok(new KickMemberResponse(true));
    }

    @PostMapping("/{teamSeq}")
    public ResponseEntity<EnterTeamResponse> enterTeam(@PathVariable Long teamSeq, @RequestBody EnterTeamRequest enterTeamRequest) {
        return ResponseEntity.ok(teamService.enterTeam(teamSeq, enterTeamRequest));
    }

    @DeleteMapping("/{teamSeq}/{userSeq}")
    public ResponseEntity<ExitTeamResponse> deleteTeam(@PathVariable Long teamSeq, @PathVariable Long userSeq) {
        teamService.exitTeam(teamSeq, userSeq);
        return ResponseEntity.ok(new ExitTeamResponse(true));
    }
}
